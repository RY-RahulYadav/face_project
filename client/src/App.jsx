import { useState } from 'react'

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadImage from './components/upload_image';
import Home from './components/home';


function App() {
  const [count, setCount] = useState(0)

  return (
    <>
     <Router>

       <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="/upload_image" element={<UploadImage />} />
          
          <Route path='*' element={<div style={{ height: '100vh' }}><h1 style={{ textAlign: 'center', position: 'relative', top: '40vh' }}>404 Page Not Found </h1></div>}> </Route>

       </Routes>
      </Router>
      
     
    </>
  )
}

export default App
