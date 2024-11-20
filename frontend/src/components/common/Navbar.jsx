import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
    const location = useLocation();

    const openFirebaseConsole = () => {
        window.open('https://console.firebase.google.com/project/face-recognition-storage/storage/face-recognition-storage.appspot.com/files/~2Fknown_people', '_blank');
    };

    return (
        <nav className="w-full bg-deep-orange-500 shadow-md">
            <div className="container mx-auto h-full w-full px-4 py-3">
                <ul className="relative flex w-full items-center justify-center space-x-10 text-9xl">
                    <li className={`navbar-item relative ${location.pathname === "/webcam" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/webcam" className="transition duration-300 hover:text-blue-200">Webcam</Link>
                    </li>
                    <li className={`navbar-item relative ${location.pathname === "/image" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/image" className="transition duration-300 hover:text-blue-200">Upload Image</Link>
                    </li>
                    <li className={`navbar-item relative ${location.pathname === "/video" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/video" className="transition duration-300 hover:text-blue-200">Upload Video</Link>
                    </li>
                    <li className="navbar-item">
                        <button
                            onClick={openFirebaseConsole}
                            className="rounded bg-white px-3 py-2 text-blue-600 transition duration-300 hover:bg-blue-100"
                        >
                            Firebase Console
                        </button>
                    </li>
                </ul>
            </div>
        </nav>
    );
};

export default Navbar;
