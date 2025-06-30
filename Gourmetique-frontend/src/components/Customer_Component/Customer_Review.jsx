import React, { useEffect, useState } from 'react';
import Customer_Nav from './Customer_Nav';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Customer_Review = () => {
  const [review, setReview] = useState([]);
  const [ads, setAds] = useState([]);
  const [hotelData, setHotelData] = useState([]);
  const [inputField, setInputField] = useState({
    image: '',
    title: '',
    hotelName: ''
  });

  const [file, setFile] = useState(null);
  const navigate = useNavigate(); // For navigation
  const customerId = sessionStorage.getItem("customerid");

  const apiLinkdisplay = 'http://localhost:3001/viewreview';
  const apiLink = 'http://localhost:3001/cusreview';
  const adsApiLink = 'http://localhost:3001/viewadmincusads';
  const apiUrlcus = 'http://localhost:3001/viewrestaurant';

  useEffect(() => {
    if (!customerId) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
    axios.get(apiLinkdisplay)
      .then((response) => {
        setReview(response.data);
        console.log(response.data);
      })
      .catch((error) => console.error('Error fetching reviews', error));

    axios.get(adsApiLink)
      .then((response) => {
        setAds(response.data);
      })
      .catch((error) => console.error('Error fetching ads', error));

    axios.get(apiUrlcus)
      .then((response) => {
        setHotelData(response.data);
      })
      .catch((error) => console.error("Error fetching restaurants:", error));
  }, [customerId, navigate]);

  const inputHandler = (event) => {
    setInputField({ ...inputField, [event.target.name]: event.target.value });
  };

  const fileHandler = (event) => {
    setFile(event.target.files[0]);
  };

  const readValue = (event) => {
    event.preventDefault();
    const formData = new FormData();

    Object.keys(inputField).forEach((key) => {
      formData.append(key, inputField[key]);
    });

    if (file) {
      formData.append('image', file);
    }

    axios.post(apiLink, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
      .then((response) => {
        if (response.data.message === 'posted successfully') {
          alert('Review posted successfully');
          setInputField({ image: '', title: '', hotelName: '' });
          setFile(null);

          axios.get(apiLinkdisplay)
            .then((response) => setReview(response.data))
            .catch((error) => console.error('Error fetching reviews', error));
        } else {
          alert('Error posting review');
        }
      })
      .catch((error) => {
        console.log(error);
        alert('Error occurred while posting review');
      });
  };

  // Styles
  const sidebarStyle = {
    width: '250px',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
  };

  const containerStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
    height: '100vh',
    backgroundColor: 'white',
    padding: '10px 10px 10px 250px',
    overflowY: 'auto',
  };

  const cardWrapperStyle = {
    width: '800px',
    backgroundColor: 'white',
    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
    borderRadius: '10px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '30px',
    overflowY: 'auto',
  };

  const cardStyle = {
    width: '100%',
    minHeight: '250px',
  };

  return (
    <div style={{ display: 'flex' }}>
      {/* Sidebar */}
      <div style={sidebarStyle}>
        <Customer_Nav />
      </div>

      {/* Card Content */}
      <div style={containerStyle}>
        <div style={cardWrapperStyle}>
          {/* Customer Reviews */}
          <div>
            <h4>Customer Reviews</h4>
            {review.length > 0 ? (
              review.map((item, index) => (
                <div key={index} className="card mb-3" style={cardStyle}>
                  <img
                    src={`http://localhost:3001/${item.image}`}
                    className="card-img-top"
                    alt="Review"
                    style={{ height: '200px', objectFit: 'cover' }}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{item.title}</h5>
                    <p className="card-text"><strong>Hotel:</strong> {item.hotelName}</p>
                  </div>
                </div>
              ))
            ) : (
              <p>No customer reviews yet.</p>
            )}
          </div>

          {/* Review Post Form */}
          <div>
            <h5>Post Your Review</h5>
            <div className="mb-2">
              <input
                onChange={fileHandler}
                accept="image/*"
                className="form-control"
                type="file"
                id="image"
                name="image"
              />
            </div>
            <form className="d-flex" onSubmit={readValue}>
              <input
                onChange={inputHandler}
                value={inputField.title}
                name="title"
                id="title"
                className="form-control me-2"
                type="text"
                placeholder="Write your review..."
              />
              <select
                name="hotelName"
                id="hotelName"
                className="form-select me-2"
                value={inputField.hotelName}
                onChange={inputHandler}
              >
                <option value="">Select Hotel</option>
                {hotelData.map((hotel, index) => (
                  <option key={index} value={hotel.restaurantName}>
                    {hotel.restaurantName}
                  </option>
                ))}
              </select>
              <button className="btn btn-outline-success" type="submit">
                POST
              </button>
            </form>
          </div>

          {/* Restaurant Ads Section */}
          <div>
            <h4>Restaurant Ads</h4>
            {ads.length > 0 ? (
              ads.map((item, index) => (
                <div key={index} className="card mb-3" style={cardStyle}>
                  <img
                    src={`http://localhost:3001/${item.image}`}
                    className="card-img-top"
                    alt="Ad"
                    style={{ height: '200px', objectFit: 'cover' }}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{item.title}</h5>
                    <p className="card-text">{item.description}</p>
                    <p className="card-text">Hotel : {item.restaurantName}</p>
                  </div>
                </div>
              ))
            ) : (
              <>
                
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Customer_Review;
