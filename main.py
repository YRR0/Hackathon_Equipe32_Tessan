'''
Main class for testing preprocessing and model pipeline
'''

import numpy as np
import torch
from pathlib import Path
from preprocessing import Preprocessor
from model import Model, MultiSpectreDataset


class MainPipeline:
    """
    Main class to orchestrate preprocessing and model testing.
    
    Allows:
    - Testing audio preprocessing (normalization, duration fix)
    - Creating spectrograms (Mel, MFCC, etc.)
    - Loading data into PyTorch DataLoaders
    - Testing model with data loaders
    """
    
    def __init__(
        self,
        data_root="data_updated",
        spectres_filename="spectres.npy",
        target_sr=22050,
        target_duration_sec=6,
        n_mels=128,
        n_mfcc=20,
        hop_length=512,
        n_fft=2048,
        batch_size=32,
        num_workers=0,
        verbose=True,
    ):
        self.data_root = Path(data_root)
        self.spectres_filename = spectres_filename
        self.target_sr = target_sr
        self.target_duration_sec = target_duration_sec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        
        # Preprocessor hyperparams
        self.preprocessor = Preprocessor(
            target_sr=target_sr,
            target_duration_sec=target_duration_sec,
            input_root="data",
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
        )
        
        # Model
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None
        
        self._log("MainPipeline initialized")
    
    def _log(self, msg):
        if self.verbose:
            print(f"[MainPipeline] {msg}")
    
    def test_preprocessing(self, input_root="data", sample_limit=3):
        """
        Test preprocessing pipeline on a sample of files.
        
        Args:
            input_root: Path to raw audio data
            sample_limit: Number of files to process for testing
        
        Returns:
            dict with preprocessing results
        """
        self._log(f"Testing preprocessing from {input_root}")
        
        from pathlib import Path
        import os
        
        input_path = Path(input_root)
        sample_count = 0
        results = {"processed": 0, "errors": 0, "files": []}
        
        for wav_file in input_path.rglob("*.wav"):
            if sample_count >= sample_limit:
                break
            
            try:
                y, sr = __import__("librosa").load(str(wav_file), sr=self.target_sr, mono=True)
                y_clean = self.preprocessor.apply_bandpass_filter(y, sr=self.target_sr)
                
                mel = self.preprocessor.compute_mel_spectrogram(y_clean)
                mfcc = self.preprocessor.compute_mfcc_spectrogram(y_clean)
                
                results["files"].append({
                    "file": wav_file.name,
                    "mel_shape": mel.shape,
                    "mfcc_shape": mfcc.shape,
                    "status": "ok"
                })
                results["processed"] += 1
                sample_count += 1
                
                self._log(f"✓ {wav_file.name}: mel{mel.shape}, mfcc{mfcc.shape}")
            
            except Exception as e:
                results["errors"] += 1
                results["files"].append({
                    "file": wav_file.name,
                    "status": "error",
                    "error": str(e)
                })
                self._log(f"✗ {wav_file.name}: {e}")
        
        self._log(f"Preprocessing test: {results['processed']} ok, {results['errors']} errors")
        return results
    
    def setup_data_loaders(self, data_root=None, feature_keys=None):
        """
        Load spectres.npy and create DataLoaders for train/val/test.
        
        Args:
            data_root: Path containing spectres.npy
            feature_keys: List of features to use (default: all available)
        
        Returns:
            (train_loader, val_loader, test_loader, dataset)
        """
        root = Path(data_root) if data_root is not None else self.data_root
        
        self._log(f"Loading data from {root / self.spectres_filename}")
        
        self.model = Model(
            data_root=str(root),
            spectres_filename=self.spectres_filename,
            feature_keys=feature_keys,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        
        self.model.load_data(data_root=str(root))
        self.train_loader, self.val_loader, self.test_loader, self.dataset = self.model.data_loader()
        
        self._log(f"DataLoaders created:")
        self._log(f"  Train batches: {len(self.train_loader)}")
        self._log(f"  Val batches: {len(self.val_loader)}")
        self._log(f"  Test batches: {len(self.test_loader)}")
        
        return self.train_loader, self.val_loader, self.test_loader, self.dataset
    
    def test_batch_loading(self, batch_idx=0):
        """
        Test loading a single batch and validate shape.
        
        Args:
            batch_idx: Which batch to load from train_loader
        
        Returns:
            dict with batch info
        """
        if self.train_loader is None:
            self._log("ERROR: DataLoaders not initialized. Call setup_data_loaders() first.")
            return None
        
        self._log(f"Testing batch loading (batch {batch_idx})...")
        
        for i, (x, y) in enumerate(self.train_loader):
            if i == batch_idx:
                info = {
                    "batch_size": x.shape[0],
                    "channels": x.shape[1],
                    "height": x.shape[2],
                    "width": x.shape[3],
                    "x_shape": tuple(x.shape),
                    "y_shape": tuple(y.shape),
                    "x_dtype": str(x.dtype),
                    "y_dtype": str(y.dtype),
                }
                self._log(f"Batch shape: {info['x_shape']}, Labels: {info['y_shape']}")
                return info
        
        self._log(f"WARNING: Batch {batch_idx} not found in train_loader")
        return None
    
    def run_full_pipeline(self, test_preprocessing=True, data_root=None, feature_keys=None):
        """
        Run complete pipeline: preprocessing test -> data loading -> batch test.
        
        Args:
            test_preprocessing: Whether to test preprocessing first
            data_root: Path to data directory
            feature_keys: Features to use
        
        Returns:
            dict with complete pipeline status
        """
        self._log("=" * 60)
        self._log("STARTING FULL PIPELINE")
        self._log("=" * 60)
        
        results = {"preprocessing": None, "data_loading": None, "batch_testing": None}
        
        # Step 1: Preprocessing test
        if test_preprocessing:
            results["preprocessing"] = self.test_preprocessing(input_root="data", sample_limit=3)
        
        # Step 2: Data loading
        try:
            self.setup_data_loaders(data_root=data_root, feature_keys=feature_keys)
            results["data_loading"] = "success"
        except Exception as e:
            self._log(f"ERROR in data loading: {e}")
            results["data_loading"] = str(e)
            return results
        
        # Step 3: Batch testing
        try:
            batch_info = self.test_batch_loading(batch_idx=0)
            results["batch_testing"] = batch_info
        except Exception as e:
            self._log(f"ERROR in batch testing: {e}")
            results["batch_testing"] = str(e)
        
        self._log("=" * 60)
        self._log("PIPELINE COMPLETE")
        self._log("=" * 60)
        
        return results


def main():
    """
    Example usage of MainPipeline for testing preprocessing and model.
    """
    
    # Initialize pipeline with custom parameters
    pipeline = MainPipeline(
        target_sr=22050,
        target_duration_sec=6,
        n_mels=128,
        n_mfcc=20,
        hop_length=512,
        n_fft=2048,
        batch_size=32,
        num_workers=0,
        verbose=True,
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        test_preprocessing=True,
        data_root=".",
        feature_keys=["mel", "mfcc", "centroid", "bandwidth", "zcr", "chroma"],
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if results["preprocessing"]:
        print(f"Preprocessing: {results['preprocessing']['processed']} files processed")
    print(f"Data Loading: {results['data_loading']}")
    if isinstance(results["batch_testing"], dict):
        print(f"Batch Shape: {results['batch_testing']['x_shape']}")


if __name__ == "__main__":
    main()