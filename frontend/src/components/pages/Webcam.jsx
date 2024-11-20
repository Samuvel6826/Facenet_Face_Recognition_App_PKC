import { useState, useEffect, useRef } from 'react';

const Webcam = () => {
    const [detectedPersons, setDetectedPersons] = useState([]);
    const previousNames = useRef(new Set());
    const eventSourceRef = useRef(null);

    // Add backend URL configuration
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:5000';

    const setupEventSource = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
        }

        // Use full URL for EventSource
        eventSourceRef.current = new EventSource(`${backendUrl}/detected_names_feed`);

        eventSourceRef.current.onmessage = (event) => {
            const names = JSON.parse(event.data);
            updateNames(names);
        };

        eventSourceRef.current.onerror = () => {
            console.error('Connection lost. Please refresh the page.');
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                // Attempt to reconnect after a delay
                setTimeout(setupEventSource, 5000);
            }
        };
    };

    const updateNames = (names) => {
        const currentNames = new Set(names);
        if (JSON.stringify([...previousNames.current]) !== JSON.stringify([...currentNames])) {
            setDetectedPersons(names);
            previousNames.current = currentNames;
        }
    };

    const handleReloadEmbeddings = async () => {
        const button = document.querySelector('.reload-button');
        if (button) {
            button.disabled = true;
            button.textContent = 'Reloading...';
        }

        try {
            // Use full URL for fetch
            const response = await fetch(`${backendUrl}/reload_embeddings`, {
                method: 'POST',
                // Add CORS headers
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.message || 'Failed to reload face data');
            alert(data.message);
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = 'Reload Face Data';
            }
        }
    };

    useEffect(() => {
        setupEventSource();
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, []);

    return (
        <div className="min-h-screen bg-gray-100">
            <header className="w-full bg-white p-5 text-center shadow-sm">
                <h1 className="text-2xl text-gray-800">Face Recognition System</h1>
                <p className="mt-1 text-gray-600">Real-time face detection and recognition</p>
            </header>

            <div className="mx-auto max-w-4xl p-4">
                <div className="rounded-lg bg-white p-6 shadow-md">
                    <div className="mb-4 flex justify-center">
                        <img
                            // Use full URL for video feed
                            src={`${backendUrl}/video_feed`}
                            alt="Video Stream"
                            className="w-full max-w-2xl rounded-lg shadow-md"
                        />
                    </div>

                    <div className="mt-4 rounded-lg bg-gray-50 p-4">
                        <h2 className="mb-3 text-center text-lg font-semibold">
                            Currently Detected People
                        </h2>
                        <div className="flex flex-wrap justify-center gap-2">
                            {detectedPersons.length > 0 ? (
                                detectedPersons.map((name, index) => (
                                    <span
                                        key={index}
                                        className="inline-flex items-center rounded-full bg-blue-100 px-3 py-1 text-blue-700"
                                    >
                                        <span className="mr-2 h-2 w-2 rounded-full bg-green-500"></span>
                                        {name}
                                    </span>
                                ))
                            ) : (
                                <span className="italic text-gray-500">
                                    No faces currently detected
                                </span>
                            )}
                        </div>
                    </div>

                    <button
                        onClick={handleReloadEmbeddings}
                        className="mt-4 w-full rounded-md bg-blue-500 px-4 py-2 text-white hover:bg-blue-600 disabled:cursor-not-allowed disabled:bg-gray-300"
                    >
                        Reload Face Data
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Webcam;