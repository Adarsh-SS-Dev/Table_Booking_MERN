import React, { useEffect, useState } from 'react';
import Restaurant_Nav from './Restaurant_Nav';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Restads = () => {
  const [vads, setVads] = useState([]);
  const [inputField, setInputField] = useState({
    image: '',
    title: '',
    description: '',
    hotelId: ''
  });
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const url = "http://localhost:3001/resads";
  const urlview = "http://localhost:3001/viewads";

  const hotelId = sessionStorage.getItem("restaurantid");
  const navigate = useNavigate();

  useEffect(() => {
    if (!hotelId) {
      navigate('/');
    } else {
      axios.post(urlview, { hotelId })
        .then((res) => setVads(res.data))
        .catch((err) => console.log('Error fetching ads:', err));
    }
  }, [hotelId, navigate]);

  const readValue = (event) => {
    event.preventDefault();
    const formdata = new FormData();

    // Manually add values
    formdata.append('title', inputField.title);
    formdata.append('description', inputField.description);
    formdata.append('hotelId', hotelId); // ✅ Correctly include hotelId

    if (file) {
      formdata.append('image', file);
    }

    axios.post(url, formdata, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then((response) => {
        if (response.data.message === 'posted successfully') {
          alert('Posted successfully');
          setInputField({ image: '', title: '', description: '', hotelId: '' });
          setFile(null);
          setPreview(null);
          axios.get(urlview)
            .then((response) => setVads(response.data))
            .catch((error) => console.error('Error fetching ads', error));
        } else {
          alert('Error posting');
        }
      })
      .catch((error) => {
        console.log(error);
        alert('Error occurred while posting ads');
      });
  };

  const fileHandler = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const inputHandler = (e) => {
    const { name, value } = e.target;
    setInputField((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const sidebarStyle = {
    width: '250px',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
  };

  const containerStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    backgroundColor: 'white',
    padding: '10px 10px 10px 250px',
  };

  const cardWrapperStyle = {
    width: '800px',
    backgroundColor: 'white',
    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
    borderRadius: '10px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflowY: 'scroll',
  };

  const cardContainerStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '20px',
    paddingBottom: '20px',
  };

  return (
    <div style={{ display: 'flex' }}>
      {/* Sidebar */}
      <div style={sidebarStyle}>
        <Restaurant_Nav />
      </div>

      {/* Main Content */}
      <div style={containerStyle}>
        <div style={cardWrapperStyle}>

          {/* View Ads */}
          <h4 className="text-center mb-3">Posted Advertisements</h4>
          <div style={cardContainerStyle}>
            {vads.length > 0 ? (
              vads.map((item, index) => (
                <div key={index} className="card" style={{ width: '100%' }}>
                  <img
                    src={`http://localhost:3001/${item.image}`}
                    className="card-img-top"
                    alt="Ad"
                    style={{ height: '200px', objectFit: 'cover' }}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{item.title}</h5>
                    <p className="card-text">{item.description}</p>
                  </div>
                </div>
              ))
            ) : (
              <p>No advertisements posted yet.</p>
            )}
          </div>

          {/* Post Ad Form */}
          <h3 className="mt-4 text-center">Post New Advertisement</h3>
          <form className="mb-3" onSubmit={readValue}>
            <input
              type="text"
              className="form-control mb-2"
              name="title"
              placeholder="Title"
              value={inputField.title}
              onChange={inputHandler}
              required
            />
            <input
              type="text"
              className="form-control mb-2"
              name="description"
              placeholder="Description"
              value={inputField.description}
              onChange={inputHandler}
              required
            />
            <input
              type="file"
              className="form-control mb-2"
              accept="image/*"
              onChange={fileHandler}
              required
            />
            {preview && (
              <img
                src={preview}
                alt="Preview"
                style={{ height: '200px', objectFit: 'cover', marginBottom: '10px' }}
              />
            )}
            <button type="submit" className="btn btn-success w-100">
              Post Advertisement
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Restads;
