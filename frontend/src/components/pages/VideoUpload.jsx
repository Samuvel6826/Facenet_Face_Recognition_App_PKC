// VideoUpload.jsx
import { useState } from 'react';
import axios from 'axios';

const VideoUpload = () => {
    const [file, setFile] = useState(null);
    const [uploadedVideoUrl, setUploadedVideoUrl] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [fileName, setFileName] = useState('No file chosen');

    const backendHostUrl = import.meta.env.VITE_BACKEND_HOST_URL || 'http://localhost:3000'; // Fallback to default URL

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            if (selectedFile.size > 10 * 1024 * 1024) { // 10 MB limit
                setError('File size exceeds 10 MB. Please select a smaller file.');
                setFileName('No file chosen');
                setFile(null);
            } else {
                setFile(selectedFile);
                setFileName(selectedFile.name);
                setError('');
            }
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setIsLoading(true);
        setUploadProgress(0);
        const formData = new FormData();
        formData.append('videoFile', file);

        try {
            await axios.post(`${backendHostUrl}/upload_video`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(percentCompleted);
                }
            }).then((response) => {
                setUploadedVideoUrl(response.data.video_url);
                setFile(null);
                setFileName('No file chosen');
            });
        } catch (error) {
            console.error('Error uploading video:', error);
            setError('Failed to upload video. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setUploadedVideoUrl('');
        setError('');
        setUploadProgress(0);
        setFileName('No file chosen');
    };

    return (
        <div className="max-w-md mx-auto p-4 border rounded-lg shadow-md bg-white">
            <h2 className="text-xl font-semibold text-center mb-4">Upload Video</h2>
            <input
                type="file"
                onChange={handleFileChange}
                accept="video/mp4,video/avi,video/mov"
                className="border rounded-lg mb-2 p-2 w-full transition duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Upload video file"
            />
            <p className="text-center mb-2 text-gray-600">{fileName}</p>
            {error && <p className="text-red-500 text-sm mb-2">{error}</p>}
            {file && (
                <div className="mb-4">
                    <h3 className="text-lg font-semibold">Selected Video:</h3>
                    <video
                        controls
                        className="mt-2 max-w-full border rounded-lg shadow-sm"
                        src={URL.createObjectURL(file)}
                    />
                </div>
            )}
            <button
                onClick={handleUpload}
                className={`bg-blue-600 text-white hover:bg-blue-700 transition duration-300 rounded-lg px-4 py-2 w-full ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isLoading || !file}
            >
                {isLoading ? 'Uploading...' : 'Upload Video'}
            </button>
            {isLoading && (
                <div className="mt-2">
                    <div className="relative pt-1">
                        <div className="flex mb-2 items-center justify-between">
                            <div>
                                <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-teal-600 bg-teal-200">
                                    Upload Progress
                                </span>
                            </div>
                            <div>
                                <span className="text-xs font-semibold inline-block text-teal-600">
                                    {uploadProgress}%
                                </span>
                            </div>
                        </div>
                        <div className="block w-full bg-gray-200 rounded">
                            <div
                                className="bg-teal-600 text-xs leading-none py-1 text-center text-white rounded"
                                style={{ width: `${uploadProgress}%` }}
                            />
                        </div>
                    </div>
                </div>
            )}
            {uploadedVideoUrl && (
                <div className="mt-4">
                    <h3 className="text-lg font-semibold">Uploaded Video:</h3>
                    <img
                        className="mt-2 max-w-full border rounded-lg shadow-sm"
                        src={`${backendHostUrl}${uploadedVideoUrl}`}
                    />
                </div>
            )}
            <button
                onClick={handleReset}
                className="bg-gray-300 text-black hover:bg-gray-400 transition duration-300 rounded-lg px-4 py-2 mt-4 w-full"
            >
                Reset
            </button>
        </div>
    );
};

export default VideoUpload;
