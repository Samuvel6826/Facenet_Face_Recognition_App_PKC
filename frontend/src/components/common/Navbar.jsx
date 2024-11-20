import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
    const location = useLocation();

    const openFirebaseConsole = () => {
        window.open('https://console.firebase.google.com/project/face-recognition-storage/storage/face-recognition-storage.appspot.com/files/~2Fknown_people', '_blank');
    };

    return (
        <nav className="bg-deep-orange-500 shadow-md w-full">
            <div className="container mx-auto px-4 py-3 w-full h-full">
                <ul className="flex items-center justify-center space-x-10 relative w-full text-9xl">
                    <li className={`navbar-item relative ${location.pathname === "/webcam" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/webcam" className="hover:text-blue-200 transition duration-300">Webcam</Link>
                    </li>
                    <li className={`navbar-item relative ${location.pathname === "/image" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/image" className="hover:text-blue-200 transition duration-300">Upload Image</Link>
                    </li>
                    <li className={`navbar-item relative ${location.pathname === "/video" ? "text-blue-200" : "text-white"}`}>
                        <Link to="/video" className="hover:text-blue-200 transition duration-300">Upload Video</Link>
                    </li>
                    <li className="navbar-item">
                        <button
                            onClick={openFirebaseConsole}
                            className="bg-white text-blue-600 hover:bg-blue-100 px-3 py-2 rounded transition duration-300"
                        >
                            Firebase Console
                        </button>
                    </li>
                </ul>
                <div className="absolute bottom-0 left-0 h-1 bg-blue-200 w-1/4 transition-all duration-300" style={{
                    transform: `translateX(${['/webcam', '/image', '/video'].indexOf(location.pathname) * 100}%)`
                }} />
            </div>
        </nav>
    );
};

export default Navbar;
