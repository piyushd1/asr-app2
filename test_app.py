import unittest
from unittest.mock import patch, MagicMock
import os
import app  # Import the application script

class TestApp(unittest.TestCase):

    def setUp(self):
        """Set up for tests."""
        # Create a dummy audio file for tests that need a filepath
        self.dummy_audio_path = "dummy_audio.wav"
        with open(self.dummy_audio_path, "w") as f:
            f.write("dummy_audio_content")

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.dummy_audio_path):
            os.remove(self.dummy_audio_path)

    def test_format_diarized_transcript(self):
        """
        Tests the formatting of the diarized transcript from a mock API response.
        """
        # Create a mock response object that mimics the Eleven Labs API response
        mock_response = MagicMock()
        mock_response.text = "Hello world this is a test."
        mock_response.words = [
            {'text': 'Hello', 'speaker_id': 'speaker_0'},
            {'text': 'world', 'speaker_id': 'speaker_0'},
            {'text': 'this', 'speaker_id': 'speaker_1'},
            {'text': 'is', 'speaker_id': 'speaker_1'},
            {'text': 'a', 'speaker_id': 'speaker_1'},
            {'text': 'test.', 'speaker_id': 'speaker_0'},
        ]

        full_transcript, diarization_info = app.format_diarized_transcript(mock_response)

        # Check formatted transcript
        self.assertIn("**Speaker 1:**", full_transcript)
        self.assertIn("**Speaker 2:**", full_transcript)
        self.assertTrue(full_transcript.startswith("**Speaker 1:** Hello world"))

        # Check diarization info
        self.assertIn("Speakers Found: 2", diarization_info)
        self.assertIn("- Speaker 1: 3 words", diarization_info)
        self.assertIn("- Speaker 2: 3 words", diarization_info)

    @patch('app.ElevenLabs')
    def test_transcribe_with_elevenlabs_success(self, mock_elevenlabs):
        """
        Tests the main transcription function on a successful API call.
        """
        # Configure the mock client and its methods
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a successful test."
        mock_response.words = [{'text': 'This', 'speaker_id': 'speaker_0'}, {'text': 'is', 'speaker_id': 'speaker_0'}, {'text': 'a', 'speaker_id': 'speaker_0'}, {'text': 'successful', 'speaker_id': 'speaker_0'}, {'text': 'test.', 'speaker_id': 'speaker_0'}]

        mock_client.speech_to_text.convert.return_value = mock_response
        mock_elevenlabs.return_value = mock_client

        full_transcript, diarization_info, status = app.transcribe_with_elevenlabs(
            api_key="fake_key",
            audio_path=self.dummy_audio_path,
            language_name="English",
            enable_diarization=True
        )

        # Assertions
        self.assertEqual(status, "✅ Transcription successful.")
        self.assertIn("**Speaker 1:**", full_transcript)
        self.assertIn("Speakers Found: 1", diarization_info)
        mock_client.speech_to_text.convert.assert_called_once()

    def test_transcribe_with_elevenlabs_no_api_key(self):
        """
        Tests that the function returns an error if no API key is provided.
        """
        full_transcript, diarization_info, status = app.transcribe_with_elevenlabs(
            api_key="",
            audio_path=self.dummy_audio_path,
            language_name="English",
            enable_diarization=True
        )
        self.assertEqual(status, "Failed")
        self.assertIn("API key is missing", full_transcript)

    @patch('app.ElevenLabs')
    def test_transcribe_with_elevenlabs_api_error(self, mock_elevenlabs):
        """
        Tests the function's error handling when the API call fails.
        """
        # Configure the mock client to raise an exception
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.side_effect = Exception("API connection failed")
        mock_elevenlabs.return_value = mock_client

        full_transcript, diarization_info, status = app.transcribe_with_elevenlabs(
            api_key="fake_key",
            audio_path=self.dummy_audio_path,
            language_name="English",
            enable_diarization=True
        )

        self.assertEqual(status, "Failed")
        self.assertIn("Error during transcription: API connection failed", full_transcript)

if __name__ == "__main__":
    unittest.main()
