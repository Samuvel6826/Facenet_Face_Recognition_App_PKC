import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/common/Navbar';
import Webcam from './components/pages/Webcam';
import ImageUpload from './components/pages/ImageUpload';
import VideoUpload from './components/pages/VideoUpload';
import './App.css';

const App = () => {
  return (
    <Router>
      <div>
        <Navbar />
        <Routes>
          <Route path="/" element={<Webcam />} />
          <Route path="/webcam" element={<Webcam />} />
          <Route path="/image" element={<ImageUpload />} />
          <Route path="/video" element={<VideoUpload />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
