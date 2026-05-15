# Enhanced UrbanInsight v3 (Khatua et al.-aligned)

## Added deliverables

1. `train_urbaninsight_v3_enhanced.py`
   - 5-city seed strategy (Vegas, Paris, Shanghai, Khartoum, Rio)
   - 70/15/15 split utilities and leakage checks
   - two-stage training: frozen backbone first 3 epochs, then unfrozen training
   - anti-overfitting hyperparameters (dropout, weight decay, mosaic/copy-paste tuning)
   - checkpointing every 5 epochs and run metadata logging (timestamp + git commit)
   - per-city validation metrics and per-city best model exports

2. `infer_with_georeferencing.py`
   - YOLOv8-seg inference
   - geospatial metadata extraction from source imagery
   - polygon georeferencing to EPSG:4326
   - JSON export with required building metadata fields
   - visualization overlay and GeoJSON export

3. `evaluate_enhanced_model.py`
   - per-city SpaceNet evaluation
   - merged SpaceNet evaluation vs paper benchmarks
   - FMOW external-domain summary metrics
   - split overlap leakage detection

4. `configs/*.yaml`
   - training hyperparameters
   - split specification template
   - model architecture settings

## Usage

```bash
python train_urbaninsight_v3_enhanced.py --help
python infer_with_georeferencing.py --help
python evaluate_enhanced_model.py --help
```

## Notes on Khatua et al. alignment

- Segmentation-first building extraction with geospatially aware output.
- Strong anti-overfitting setup and leakage checks.
- External-domain testing path (FMOW) without mixing FMOW into training.
