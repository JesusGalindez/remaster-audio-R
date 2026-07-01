import os
import sys
import subprocess
import unittest
from unittest.mock import MagicMock, patch
import remasterer

class TestRemasterer(unittest.TestCase):
    
    def setUp(self):
        self.input_mp4 = "test_input.mp4"
        self.output_mp4 = "test_output.mp4"
        self.ref_wav = "test_ref.wav"
        
        # Clean up previous tests
        self.tearDown()
        
    def tearDown(self):
        for f in [self.input_mp4, self.output_mp4, self.ref_wav]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
                    
    def test_ffmpeg_and_model_download(self):
        # 1. Test FFmpeg detection/download
        ffmpeg_bin = remasterer.get_ffmpeg_binary()
        self.assertTrue(os.path.exists(ffmpeg_bin))
        self.assertTrue(os.access(ffmpeg_bin, os.X_OK))
        
        # 2. Test Model download
        model_path = remasterer.get_rnnoise_model()
        self.assertTrue(os.path.exists(model_path))
        
    def test_full_remastering_pipeline(self):
        ffmpeg_bin = remasterer.get_ffmpeg_binary()
        
        # Create a dummy 5-second MP4 video file with a test audio tone
        cmd_create = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=48000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            self.input_mp4
        ]
        result = subprocess.run(cmd_create, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, f"Failed to create dummy video: {result.stderr.decode()}")
        self.assertTrue(os.path.exists(self.input_mp4))
        
        # Run processing on the dummy video
        print("Running remasterer on dummy input video...")
        success = remasterer.process_file(self.input_mp4, self.output_mp4, sync_ms=100)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.output_mp4))
        self.assertGreater(os.path.getsize(self.output_mp4), 0)

    def test_ai_lip_sync_pipeline_mock(self):
        # We will mock SyncNet imports and pipeline inside our test so it doesn't
        # try to download PyTorch and weights during automated tests.
        ffmpeg_bin = remasterer.get_ffmpeg_binary()
        
        # Create dummy video
        cmd_create = [
            ffmpeg_bin, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=48000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            self.input_mp4
        ]
        subprocess.run(cmd_create, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Verify metadata extraction helpers
        duration = remasterer.get_video_duration(self.input_mp4, ffmpeg_bin)
        self.assertAlmostEqual(duration, 5.0, places=1)
        
        fps = remasterer.get_video_fps(self.input_mp4, ffmpeg_bin)
        self.assertAlmostEqual(fps, 10.0, places=1)
        
        # Mock sys.modules for syncnet_python so the patch look-up succeeds
        mock_syncnet_module = MagicMock()
        sys.modules['syncnet_python'] = mock_syncnet_module
        
        # Mock SyncNet call
        mock_pipeline = MagicMock()
        mock_pipeline.inference.return_value = ([2], [8.5], [0.3], 8.5, 0.3, "{}", True)
        mock_syncnet_module.SyncNetPipeline = MagicMock(return_value=mock_pipeline)
        
        with patch('remasterer.ensure_syncnet_installed') as mock_install, \
             patch('remasterer.download_syncnet_weights', return_value=('sfd.pth', 'sync.model')) as mock_weights:
            
            # Run processing with AI Sync
            success = remasterer.process_file(self.input_mp4, self.output_mp4, auto_sync_lips=True)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(self.output_mp4))
            self.assertGreater(os.path.getsize(self.output_mp4), 0)

if __name__ == "__main__":
    unittest.main()
