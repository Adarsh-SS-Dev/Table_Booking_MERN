import React, { useEffect, useState } from 'react';
import Adminnav from './Adminnav';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Review_Ads = () => {
  const [review, setReview] = useState([]);
  const [ads, setAds] = useState([]);
  const apiLink = "http://localhost:3001/viewreview";
  const urlview = "http://localhost:3001/viewadmincusads";
  const admtoken = sessionStorage.getItem("adtoken");
  const navigate = useNavigate();

  useEffect(() => {
    if (!admtoken) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
    axios.get(apiLink)
      .then((response) => {
        setReview(response.data);
        console.log("Reviews:", response.data);
      });

    axios.get(urlview)
      .then((response) => {
        setAds(response.data);
        console.log("Ads:", response.data);
      });
  }, [admtoken, navigate]);

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
    height: '700px',
    backgroundColor: 'white',
    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
    borderRadius: '10px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
    gap: '20px',
  };

  const cardStyle = {
    border: '1px solid #ccc',
    borderRadius: '10px',
    padding: '15px',
    boxShadow: '0px 2px 6px rgba(0, 0, 0, 0.05)',
    backgroundColor: '#f9f9f9',
  };

  return (
    <div style={{ display: 'flex' }}>
      {/* Sidebar */}
      <div style={sidebarStyle}>
        <Adminnav />
      </div>

      {/* Content */}
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
              <p>No reviews available.</p>
            )}
          </div>

          {/* Restaurant Ads Card */}
          <div style={cardStyle}>
            <h4>Restaurant Advertisements</h4>

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
                    <p>{item.description}</p>
                    <p>Hotel : {item.restaurantName}</p>
                  </div>
                </div>
              ))
            ) : (
              <p>No advertisements available.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Review_Ads;
