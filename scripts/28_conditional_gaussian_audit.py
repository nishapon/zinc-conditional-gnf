#!/usr/bin/env python3
"""Read-only audit of the *trained* N+QED GNF's conditional diagonal Gaussian.

For real held-out molecular embeddings, calculate z = flow(encoder(graph), c),
then standardized residuals r = (z - prior_mu(c)) / prior_sigma(c).
If the fitted conditional prior is well calibrated, r should approximately
follow independent standard normal coordinates (with caveats about dependence
across atoms in the same molecule and finite samples).

Uses the VALIDATION split only. No checkpoints are written or updated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from zinc_gnf.conditioning import batch_condition, condition_embedding_for
from zinc_gnf.dataset import MolecularGraphDataset, iter_eval_batches, load_training_splits
from zinc_gnf.trainer_autoencoder import load_autoencoder
from zinc_gnf.trainer_flow import file_sha256, load_flow


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow-checkpoint', default='outputs/checkpoints/warm_start_stageA/last.pt')
    p.add_argument('--ae-checkpoint', default=None)
    p.add_argument('--data-dir', default='data/processed/zinc250k_charge_aware_moflow_fixed')
    p.add_argument('--output-dir', default='outputs/metrics/conditional_gaussian_audit')
    p.add_argument('--global-molecules', type=int, default=256)
    p.add_argument('--per-cell-molecules', type=int, default=32)
    p.add_argument('--n-values', default='15,23,30')
    p.add_argument('--qed-centers', default='0.4,0.6,0.8')
    p.add_argument('--qed-halfwidth', type=float, default=0.075)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=1729)
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--no-smoothing', action='store_true',
                   help='Diagnostic alternate: omit smoothing noise used during training')
    return p.parse_args()


def summarize(r: np.ndarray, sigma: np.ndarray, *, molecules: int, name: str) -> dict:
    """r: all valid-node whitened residuals, shape [nodes, dim]."""
    if len(r) == 0:
        return {'group': name, 'molecules': molecules, 'nodes': 0, 'note': 'empty'}
    residual = np.asarray(r, dtype=np.float64)
    if not np.isfinite(residual).all():
        return {'group': name, 'molecules': molecules, 'nodes': len(r),
                'error': 'non-finite normalized latent residuals'}
    mean = residual.mean(axis=0)
    std = residual.std(axis=0)
    centred = residual - mean
    good = std > 1e-8
    skew = np.full(len(std), np.nan)
    kurt = np.full(len(std), np.nan)
    if np.any(good):
        standardized = centred[:, good] / std[good]
        skew[good] = np.mean(standardized**3, axis=0)
        kurt[good] = np.mean(standardized**4, axis=0) - 3
    # Correlations between latent dimensions; for a diagonal Gaussian these
    # should be small, though high-dimensional sample noise limits precision.
    if np.sum(good) >= 2:
        corr = np.corrcoef(residual[:, good], rowvar=False)
        offdiag = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
        max_corr = float(np.max(offdiag))
        mean_corr = float(np.mean(offdiag))
    else:
        max_corr = float('nan')
        mean_corr = float('nan')
    return {
        'group': name, 'molecules': int(molecules), 'nodes': int(len(residual)),
        'dimensions': int(residual.shape[1]),
        'mean_abs_standardized_mean': float(np.mean(np.abs(mean))),
        'rms_standardized_mean': float(np.sqrt(np.mean(mean**2))),
        'rms_standardized_residual': float(np.sqrt(np.mean(residual**2))),
        'median_empirical_std': float(np.median(std)),
        'p95_abs_standardized_residual': float(np.quantile(np.abs(residual), .95)),
        'within_1_96_fraction': float(np.mean(np.abs(residual) <= 1.96)),
        'above_3_fraction': float(np.mean(np.abs(residual) > 3)),
        'median_abs_skew': float(np.nanmedian(np.abs(skew))),
        'median_excess_kurtosis': float(np.nanmedian(kurt)),
        'max_abs_dimension_correlation': max_corr,
        'mean_abs_dimension_correlation': mean_corr,
        'mean_predicted_sigma': float(np.mean(sigma)),
    }


@torch.no_grad()
def audit_one(indices, dataset, autoencoder, flow, *, device, batch_size, seed,
              smoothing, group_name):
    residuals = []
    sigmas = []
    generator = torch.Generator(device=device).manual_seed(seed)
    finite_failures = 0
    good_molecules = 0
    for batch in iter_eval_batches(dataset, batch_size, indices):
        batch = batch.to(device)
        h = autoencoder.encode(batch)
        if smoothing:
            h = flow.add_smoothing_noise(h, batch.node_mask, generator=generator)
        values, present = batch_condition(batch, flow.config.condition_features)
        condition = condition_embedding_for(flow, values, present, batch.batch_size, device)
        z, _ = flow(h, batch.node_mask, condition)
        mu, sigma = flow.prior.parameters_for(batch.batch_size, condition, device)
        r = (z - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        finite = (torch.isfinite(r) | ~batch.node_mask.unsqueeze(-1)).all(dim=(1,2))
        finite_failures += int((~finite).sum().item())
        good_molecules += int(finite.sum().item())
        if finite.any():
            keep_nodes = batch.node_mask & finite.unsqueeze(1)
            residuals.append(r[keep_nodes].detach().cpu().numpy())
            sigmas.append(sigma[finite].detach().cpu().numpy())
    dim = flow.config.node_embedding_dim
    res = np.concatenate(residuals, axis=0) if residuals else np.empty((0, dim))
    sigma_values = np.concatenate(sigmas, axis=0) if sigmas else np.empty((0, dim))
    summary = summarize(res, sigma_values, molecules=good_molecules, name=group_name)
    summary['requested_molecules'] = len(indices)
    summary['nonfinite_molecules'] = finite_failures
    return summary, res


def main():
    args = parse_args()
    if args.batch_size < 1 or args.global_molecules < 1 or args.per_cell_molecules < 1:
        raise ValueError('batch size and molecular counts must be positive')
    if args.qed_halfwidth <= 0:
        raise ValueError('qed halfwidth must be positive')
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    device = torch.device(args.device)
    start = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    flow_path = Path(args.flow_checkpoint)
    flow, meta = load_flow(flow_path, device=device)
    if not set(('n','qed')).issubset(flow.config.condition_features):
        raise ValueError(f'expected a jointly N+QED-conditional checkpoint; got {flow.config.condition_features}')
    recorded_ae = meta.get('autoencoder') or {}
    ae_path = Path(args.ae_checkpoint or recorded_ae.get('checkpoint', ''))
    if not ae_path.is_file():
        raise FileNotFoundError(f'autoencoder checkpoint missing: {ae_path}; use --ae-checkpoint')
    recorded_hash = recorded_ae.get('sha256')
    actual_hash = file_sha256(ae_path)
    if recorded_hash and recorded_hash != actual_hash:
        raise ValueError('autoencoder checkpoint SHA256 differs from the flow training provenance')
    autoencoder, _ = load_autoencoder(ae_path, device=device)
    autoencoder.eval()
    flow.eval()
    for model in (autoencoder, flow):
        model.requires_grad_(False)
    if flow.config.node_embedding_dim != autoencoder.config.node_embedding_dim:
        raise ValueError('autoencoder and flow embedding dimension mismatch')
    train, val, fingerprint = load_training_splits(args.data_dir)
    del train
    dataset = MolecularGraphDataset(val)
    del val
    rng = np.random.default_rng(args.seed)
    n_values = [int(x.strip()) for x in args.n_values.split(',') if x.strip()]
    qed_centers = [float(x.strip()) for x in args.qed_centers.split(',') if x.strip()]
    global_indices = rng.choice(len(dataset), size=min(args.global_molecules,len(dataset)),
                                replace=False).tolist()
    sets = [('random_validation', global_indices)]
    for n in n_values:
        for q in qed_centers:
            eligible = np.flatnonzero((dataset.n_node == n) &
                            (np.abs(dataset.qed - q) <= args.qed_halfwidth))
            chosen = rng.choice(eligible, size=min(args.per_cell_molecules,len(eligible)),
                                replace=False).tolist() if len(eligible) else []
            sets.append((f'N={n},QED={q:.2f}+/-{args.qed_halfwidth:g} (eligible={len(eligible)})',chosen))
    print('FLOW:',flow_path,'step:',meta.get('step'),'AE:',ae_path, flush=True)
    print('Validation molecules:',len(dataset), 'smoothing:',not args.no_smoothing,
          'noise_sigma:',flow.config.noise_sigma, flush=True)
    summaries = []
    pooled_residuals = None
    for ordinal,(name,indices) in enumerate(sets):
        if not indices:
            summaries.append({'group':name,'molecules':0,'nodes':0,'note':'no validation support'})
            print(name,': no validation support',flush=True)
            continue
        result, r = audit_one(indices,dataset,autoencoder,flow,device=device,
                     batch_size=args.batch_size,seed=args.seed+ordinal,
                     smoothing=not args.no_smoothing,group_name=name)
        summaries.append(result)
        if ordinal == 0:
            pooled_residuals = r
        print(name, '| molecules:',result['molecules'], '| z RMS:',
              round(result.get('rms_standardized_residual',float('nan')),3),
              '| 95% coverage:',round(result.get('within_1_96_fraction',float('nan')),3),
              '| corr:',round(result.get('mean_abs_dimension_correlation',float('nan')),3),
              '| failed:',result['nonfinite_molecules'],flush=True)
    report = {
        'purpose': 'Read-only held-out conditional latent Gaussian calibration audit',
        'interpretation': ('Standard normal targets for pooled per-node residuals: '
              'RMS=1, |r|<=1.96 coverage ~0.95, |r|>3 fraction ~0.0027, '
              'mean/skew/excess-kurtosis ~0, dimension correlations ~0. '
              'Nodes within a molecule are dependent; these are diagnostics, not p-values. '
              'Even a mismatch does not prove that the Gaussian family is intrinsically wrong: '
              'the flow may be undertrained or have insufficient capacity.'),
        'reference_gaussian': {'rms':1.0,'coverage95':0.95,'above3':0.0026998,
                              'excess_kurtosis':0.0},
        'flow_checkpoint':str(flow_path), 'flow_sha256':file_sha256(flow_path),
        'flow_step':meta.get('step'),'flow_config':flow.config.to_dict(),
        'ae_checkpoint':str(ae_path),'ae_sha256':actual_hash,
        'data_dir':args.data_dir,'data_fingerprint':fingerprint,
        'validation_molecules':len(dataset),'smoothing':not args.no_smoothing,
        'seed':args.seed,'elapsed_seconds':time.perf_counter()-start,
        'groups':summaries}
    (out/'gaussian_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str))
    fields = sorted(set().union(*(set(s.keys()) for s in summaries)))
    with (out/'gaussian_audit.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    lines = [f'CONDITIONAL LATENT GAUSSIAN AUDIT | flow step {meta.get("step")}',
             f'Validation: {len(dataset)} molecules | smoothing={not args.no_smoothing}',
             'Reference: z-score RMS 1, coverage ±1.96 0.95, P(|z|>3) 0.0027, zero kurtosis.',
             'No p-values: node samples inside a molecule are dependent.', '']
    for row in summaries:
        lines += [row['group'],
            f"  sampled {row.get('molecules',0)} molecules, {row.get('nodes',0)} atoms; "
            f"nonfinite {row.get('nonfinite_molecules',0)}"]
        if 'rms_standardized_residual' in row:
            lines += [f"  standardized RMS: {row['rms_standardized_residual']:.3f}",
               f"  coverage ±1.96: {row['within_1_96_fraction']:.3f}",
               f"  P(|r| > 3): {row['above_3_fraction']:.4f}",
               f"  median excess kurtosis: {row['median_excess_kurtosis']:+.3f}",
               f"  mean abs cross-dim corr: {row['mean_abs_dimension_correlation']:.3f}"]
        elif 'note' in row:
            lines.append('  '+row['note'])
        lines.append('')
    (out/'gaussian_audit.txt').write_text('\n'.join(lines))
    # Simple reference plot using a seeded subsample of residual coordinates.
    if pooled_residuals is not None and pooled_residuals.size:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        flat = pooled_residuals.ravel()
        if len(flat)>100000:
            flat = flat[np.random.default_rng(args.seed).choice(len(flat),100000,replace=False)]
        x = np.linspace(-4.5,4.5,400)
        fig,ax=plt.subplots(figsize=(7,4.2))
        ax.hist(flat,bins=90,range=(-4.5,4.5),density=True,alpha=.60,label='held-out residuals')
        ax.plot(x,np.exp(-x*x/2)/np.sqrt(2*np.pi),lw=2,label='N(0,1) reference')
        ax.set(xlabel='(z - predicted mu) / predicted sigma', ylabel='Density',
               title='Conditional latent Gaussian: held-out validation')
        ax.legend(); fig.tight_layout()
        fig.savefig(out/'standardized_residuals.png',dpi=160)
        plt.close(fig)
    print('Completed:',out/'gaussian_audit.txt',flush=True)


if __name__ == '__main__':
    main()
