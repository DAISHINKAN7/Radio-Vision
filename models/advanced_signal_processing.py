"""
Advanced Signal Processing for Radio Vision
- FITS file support
- RFI removal
- Wavelet decomposition
- Advanced filtering
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import zoom

class AdvancedSignalProcessor:
    """Advanced signal processing utilities"""
    
    def __init__(self):
        self.fits_available = self._check_fits()
        self.wavelet_available = self._check_wavelet()
    
    def _check_fits(self):
        """Check if astropy is available"""
        try:
            from astropy.io import fits
            self.fits = fits
            return True
        except ImportError:
            print("⚠️  FITS support requires: pip install astropy")
            return False
    
    def _check_wavelet(self):
        """Check if PyWavelets is available"""
        try:
            import pywt
            self.pywt = pywt
            return True
        except ImportError:
            print("⚠️  Wavelet support requires: pip install PyWavelets")
            return False
    
    # ============================================================================
    # FITS FILE HANDLING
    # ============================================================================
    
    def load_fits(self, filepath):
        """
        Load signal from FITS file
        
        Args:
            filepath: Path to FITS file
        
        Returns:
            numpy array
        """
        if not self.fits_available:
            raise ImportError("FITS support requires: pip install astropy")
        
        try:
            hdu = self.fits.open(filepath)
            data = hdu[0].data
            
            # Handle different FITS structures
            if 'DATA' in data.dtype.names if hasattr(data, 'dtype') else []:
                # Visibility data
                visibilities = data['DATA']
                signal = self._visibilities_to_signal(visibilities)
            else:
                # Direct image data
                signal = np.array(data, dtype=np.float32)
            
            hdu.close()
            
            # Ensure 2D
            if signal.ndim > 2:
                signal = signal[0] if signal.shape[0] == 1 else signal.mean(axis=0)
            elif signal.ndim == 1:
                signal = signal.reshape(-1, 1)
            
            return signal
        
        except Exception as e:
            raise ValueError(f"Failed to load FITS file: {e}")
    
    def _visibilities_to_signal(self, visibilities):
        """Convert visibility data to signal spectrogram"""
        # Take magnitude
        if np.iscomplexobj(visibilities):
            magnitude = np.abs(visibilities)
        else:
            magnitude = visibilities
        
        # Reshape to 2D
        if magnitude.ndim > 2:
            magnitude = magnitude.reshape(-1, magnitude.shape[-1])
        
        return magnitude.astype(np.float32)
    
    # ============================================================================
    # RFI (RADIO FREQUENCY INTERFERENCE) REMOVAL
    # ============================================================================
    
    def remove_rfi(self, signal, threshold_sigma=3.0, method='statistical'):
        """
        Remove RFI from signal
        
        Args:
            signal: numpy array (H, W)
            threshold_sigma: sigma threshold for outlier detection
            method: 'statistical', 'morphological', or 'spectral'
        
        Returns:
            Cleaned signal
        """
        if method == 'statistical':
            return self._remove_rfi_statistical(signal, threshold_sigma)
        elif method == 'morphological':
            return self._remove_rfi_morphological(signal)
        elif method == 'spectral':
            return self._remove_rfi_spectral(signal, threshold_sigma)
        else:
            raise ValueError(f"Unknown RFI removal method: {method}")
    
    def _remove_rfi_statistical(self, signal, threshold_sigma):
        """Statistical RFI removal using sigma clipping"""
        
        # Calculate statistics
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        std = mad * 1.4826  # Robust std estimate
        
        # Find outliers
        z_scores = np.abs((signal - median) / std)
        mask = z_scores > threshold_sigma
        
        # Replace outliers with median
        cleaned = signal.copy()
        cleaned[mask] = median
        
        return cleaned
    
    def _remove_rfi_morphological(self, signal):
        """Morphological RFI removal"""
        from scipy.ndimage import median_filter, binary_dilation
        
        # Apply median filter
        background = median_filter(signal, size=5)
        
        # Find spikes
        diff = np.abs(signal - background)
        threshold = np.percentile(diff, 95)
        spikes = diff > threshold
        
        # Dilate spike mask
        spikes_dilated = binary_dilation(spikes, iterations=2)
        
        # Replace spikes
        cleaned = signal.copy()
        cleaned[spikes_dilated] = background[spikes_dilated]
        
        return cleaned
    
    def _remove_rfi_spectral(self, signal, threshold_sigma):
        """Spectral RFI removal using FFT"""
        
        # FFT
        fft = np.fft.fft2(signal)
        fft_mag = np.abs(fft)
        
        # Find outliers in frequency domain
        median = np.median(fft_mag)
        mad = np.median(np.abs(fft_mag - median))
        std = mad * 1.4826
        
        threshold = median + threshold_sigma * std
        mask = fft_mag > threshold
        
        # Remove outliers
        fft_cleaned = fft.copy()
        fft_cleaned[mask] = median * np.exp(1j * np.angle(fft[mask]))
        
        # Inverse FFT
        cleaned = np.real(np.fft.ifft2(fft_cleaned))
        
        return cleaned
    
    # ============================================================================
    # WAVELET DECOMPOSITION
    # ============================================================================
    
    def wavelet_decompose(self, signal, wavelet='db4', level=3):
        """
        Wavelet decomposition for multi-scale analysis
        
        Args:
            signal: numpy array (H, W)
            wavelet: wavelet type ('db4', 'haar', 'sym4', etc.)
            level: decomposition level
        
        Returns:
            coefficients: list of (cA, (cH, cV, cD)) tuples
        """
        if not self.wavelet_available:
            raise ImportError("Wavelet support requires: pip install PyWavelets")
        
        try:
            coeffs = self.pywt.wavedec2(signal, wavelet, level=level)
            return coeffs
        except Exception as e:
            raise ValueError(f"Wavelet decomposition failed: {e}")
    
    def wavelet_denoise(self, signal, wavelet='db4', threshold_type='soft'):
        """
        Denoise signal using wavelet thresholding
        
        Args:
            signal: numpy array (H, W)
            wavelet: wavelet type
            threshold_type: 'soft' or 'hard'
        
        Returns:
            Denoised signal
        """
        if not self.wavelet_available:
            return signal  # Fallback: return original
        
        # Decompose
        coeffs = self.pywt.wavedec2(signal, wavelet, level=3)
        
        # Estimate noise std
        cH1 = coeffs[-1][0]  # First level horizontal detail
        sigma = np.median(np.abs(cH1)) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(signal.size))
        
        # Threshold detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation
        for detail_coeffs in coeffs[1:]:
            new_detail = []
            for coef in detail_coeffs:
                if threshold_type == 'soft':
                    thresholded = self.pywt.threshold(coef, threshold, mode='soft')
                else:
                    thresholded = self.pywt.threshold(coef, threshold, mode='hard')
                new_detail.append(thresholded)
            new_coeffs.append(tuple(new_detail))
        
        # Reconstruct
        denoised = self.pywt.waverec2(new_coeffs, wavelet)
        
        # Handle size mismatch
        if denoised.shape != signal.shape:
            denoised = denoised[:signal.shape[0], :signal.shape[1]]
        
        return denoised
    
    # ============================================================================
    # ADVANCED FILTERING
    # ============================================================================
    
    def baseline_subtraction(self, signal, method='polynomial', order=2):
        """
        Remove baseline from signal
        
        Args:
            signal: numpy array (H, W)
            method: 'polynomial', 'rolling', or 'als'
            order: polynomial order (for polynomial method)
        
        Returns:
            Baseline-subtracted signal
        """
        if method == 'polynomial':
            return self._baseline_polynomial(signal, order)
        elif method == 'rolling':
            return self._baseline_rolling(signal)
        elif method == 'als':
            return self._baseline_als(signal)
        else:
            return signal
    
    def _baseline_polynomial(self, signal, order):
        """Polynomial baseline subtraction"""
        # Fit polynomial along each row
        corrected = np.zeros_like(signal)
        
        for i in range(signal.shape[0]):
            x = np.arange(signal.shape[1])
            coeffs = np.polyfit(x, signal[i], order)
            baseline = np.polyval(coeffs, x)
            corrected[i] = signal[i] - baseline
        
        return corrected
    
    def _baseline_rolling(self, signal, window=51):
        """Rolling median baseline subtraction"""
        from scipy.ndimage import median_filter
        
        baseline = median_filter(signal, size=window)
        return signal - baseline
    
    def _baseline_als(self, signal, lam=1e6, p=0.01, niter=10):
        """Asymmetric Least Squares baseline"""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        L = signal.shape[1]
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * signal[0])
            w = p * (signal[0] > z) + (1-p) * (signal[0] < z)
        
        baseline = np.tile(z, (signal.shape[0], 1))
        return signal - baseline
    
    def frequency_calibration(self, signal, freq_range=None):
        """
        Calibrate frequency axis
        
        Args:
            signal: numpy array (H, W)
            freq_range: tuple (min_freq, max_freq) in Hz
        
        Returns:
            Calibrated signal
        """
        # For now, just normalize
        # In production, would apply actual frequency calibration
        return (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    
    # ============================================================================
    # SIGNAL QUALITY METRICS
    # ============================================================================
    
    def calculate_snr(self, signal):
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(signal ** 2)
        noise = signal - sp_signal.medfilt2d(signal, kernel_size=5)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0
        
        return float(snr)
    
    def calculate_dynamic_range(self, signal):
        """Calculate dynamic range"""
        max_val = np.max(signal)
        min_val = np.min(signal[signal > 0]) if np.any(signal > 0) else np.min(signal)
        
        if min_val > 0:
            dr = 20 * np.log10(max_val / min_val)
        else:
            dr = 100.0
        
        return float(dr)
    
    def calculate_spectral_purity(self, signal):
        """Calculate spectral purity"""
        fft = np.fft.fft2(signal)
        power_spectrum = np.abs(fft) ** 2
        
        # Find peak
        peak_power = np.max(power_spectrum)
        total_power = np.sum(power_spectrum)
        
        purity = peak_power / total_power if total_power > 0 else 0
        
        return float(purity)


# Usage example
if __name__ == "__main__":
    print("Testing Advanced Signal Processor...")
    
    processor = AdvancedSignalProcessor()
    
    # Test with random signal
    test_signal = np.random.rand(128, 1024).astype(np.float32)
    
    # Add some RFI
    test_signal[50:55, 400:450] = 5.0  # Spike
    
    print("\n1. RFI Removal:")
    cleaned = processor.remove_rfi(test_signal, threshold_sigma=3.0, method='statistical')
    print(f"   Original max: {test_signal.max():.2f}")
    print(f"   Cleaned max: {cleaned.max():.2f}")
    
    print("\n2. Signal Quality:")
    snr = processor.calculate_snr(test_signal)
    print(f"   SNR: {snr:.2f} dB")
    
    dr = processor.calculate_dynamic_range(test_signal)
    print(f"   Dynamic Range: {dr:.2f} dB")
    
    print("\n3. Wavelet Denoising:")
    if processor.wavelet_available:
        denoised = processor.wavelet_denoise(test_signal)
        print(f"   ✅ Denoised shape: {denoised.shape}")
    else:
        print(f"   ⚠️  Install PyWavelets for wavelet support")
    
    print("\n4. FITS Support:")
    if processor.fits_available:
        print("   ✅ FITS files supported")
    else:
        print("   ⚠️  Install astropy for FITS support")
    
    print("\n✅ All tests complete!")