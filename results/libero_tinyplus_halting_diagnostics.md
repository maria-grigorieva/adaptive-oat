# Tinyplus Halting Diagnostics
- Dataset: `data/libero/libero_spatial_tinyplus.zarr`
- Tokenizer checkpoint: `/Users/maria/MIPT/AOPT_assignment/adaptive-oat/output/20260428/232508_train_oattok_tinyplus_cpu_libero_spatial_tinyplus/checkpoints/latest.ckpt`
- Action chunks analyzed: `2356`
## Answers
1. Oracle K distribution mean: `2.2602`.
   Oracle distribution over `[1, 2, 4, 8]`: `{1: 0.7966893039049237, 2: 0.0050933786078098476, 4: 0.03310696095076401, 8: 0.16511035653650255}`.
2. Adaptive predicted K mean: `1.1538`.
   Predicted distribution over `1..8`: `{1: 0.8461538461538461, 2: 0.15384615384615385, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}`.
3. Adaptive halting collapsed toward early stopping: `True`.
   Oracle deep-prefix mass (`K>=4`) is `0.1982`.
4. Recommended next changes:
- Train the adaptive policy longer before comparing control behavior.
- Run evaluation with `--adaptive-halting --adaptive-min-k 2` as a conservative calibration baseline.
- Consider a minimum prefix depth during early training or a penalty against EOS before K=2.
- Tune `halt_tolerance` and budget regularization so oracle supervision is less aggressively biased toward shallow prefixes if needed.
## Reconstruction Proxy
- `fixed_k_1`: avg_K=`1.0000`, token_ratio=`0.1250`, recon_mse=`0.266260`
- `fixed_k_2`: avg_K=`2.0000`, token_ratio=`0.2500`, recon_mse=`0.266280`
- `fixed_k_4`: avg_K=`4.0000`, token_ratio=`0.5000`, recon_mse=`0.266304`
- `fixed_k_8`: avg_K=`8.0000`, token_ratio=`1.0000`, recon_mse=`0.266326`
- `oracle_k`: avg_K=`2.2602`, token_ratio=`0.2825`, recon_mse=`0.265960`
- `adaptive_predicted_k`: avg_K=`1.1538`, token_ratio=`0.1442`, recon_mse=`0.266263`