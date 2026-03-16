import sys
from pathlib import Path
from unittest import mock

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tests.mock_gpu  # noqa: F401
import shorts  # noqa: E402

@mock.patch("shorts.process_video")
@mock.patch("shorts.Path.iterdir")
@mock.patch("shorts.Path.exists")
def test_main(mock_exists, mock_iterdir, mock_process_video):
    # Setup gameplay dir mock
    mock_exists.return_value = True
    
    # Setup some file mocks
    mock_file1 = mock.MagicMock()
    mock_file1.is_file.return_value = True
    mock_file1.suffix = ".mp4"
    
    mock_file2 = mock.MagicMock()
    mock_file2.is_file.return_value = True
    mock_file2.suffix = ".txt"
    
    # Set iterdir to return these two files
    mock_iterdir.return_value = [mock_file1, mock_file2]
    
    shorts.main()
    
    # process_video should only be called once, for the .mp4 file
    mock_process_video.assert_called_once()
    assert mock_process_video.call_args[0][0] == mock_file1
