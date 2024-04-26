import React, { useEffect, useState } from "react";

function UploadImage() {
    const [count, setCount] = useState(0);
    const [selectedImage, setSelectedImage] = useState([]);
    useEffect(() => {}, [count]);

   function handleclick(){
        setCount(count+1)
    }


    const handleChange = (e) => {
        const files = Array.from(e.target.files);
    
        const formData = new FormData();
    
        files.forEach((file, index) => {
          formData.append(`image${index}`, file);
        });
    
        setSelectedImage(formData);
      };


    const handleSubmit = async (event) => {
        event.preventDefault();
        fetch('http://127.0.0.1:5000/upload_file', {
            method: 'POST',
            body: selectedImage
         })
      .then(response => response.json())
      .then(data => {
        console.log(data);
      })
      .catch(error => {
        console.error('Error uploading images:', error);
      });
    
    }

    

    return <div>
    <form action="" method="POST">
    <input onChange={handleChange} type="file"  name ='image'  accept=".jpg, .jpeg, .png" multiple/>

    

    <button  onClick ={handleSubmit} type="submit">submit</button></form>
   
  </div>
}

export default UploadImage


   