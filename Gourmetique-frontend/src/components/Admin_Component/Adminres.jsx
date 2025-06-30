import React, { useEffect, useState } from 'react';
import Adminnav from './Adminnav';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Adminres = () => {
  const [restaurants, setRestaurants] = useState([]);

  const apiUrl = "http://localhost:3001/viewrestaurant";
  const apiUrl1 = "http://localhost:3001/delres";
  const admtoken = sessionStorage.getItem("adtoken");
  const navigate = useNavigate();
  

  useEffect(() => {
    if (!admtoken) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
    axios
      .get(apiUrl)
      .then((response) => {
        setRestaurants(response.data);
      })
      .catch((error) => console.error("Error fetching restaurants:", error));
  }, [admtoken, navigate]);

  const handleDelete = async (id) => {
    try {
      const response = await axios.post(apiUrl1, { _id: id });
      if (response.status === 200) {
        // Remove the deleted restaurant from the UI
        setRestaurants((prev) => prev.filter((res) => res._id !== id));
        alert("Restaurant deleted successfully!");
      } else {
        alert("Failed to delete restaurant.");
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Error deleting restaurant.");
    }
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
    height: '700px',
    backgroundColor: 'white',
    boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
    borderRadius: '10px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto',
  };

  const cardStyle = {
    border: '1px solid #ddd',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '15px',
    boxShadow: '0px 2px 5px rgba(0,0,0,0.1)',
  };

  const buttonStyle = {
    marginTop: '10px',
    backgroundColor: '#dc3545',
    color: 'white',
    border: 'none',
    padding: '6px 12px',
    borderRadius: '5px',
    cursor: 'pointer',
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
          <h3>Restaurant Details</h3>
          {restaurants.length > 0 ? (
            restaurants.map((res) => (
              <div key={res._id} style={cardStyle}>
                <p><strong>Name:</strong> {res.restaurantName}</p>
                <p><strong>Owned By:</strong> {res.ownerName}</p>
                <p><strong>Email:</strong> {res.email}</p>
                <p><strong>Phone:</strong> {res.phone}</p>
                <p><strong>Address:</strong> {res.address}</p>
                <p><strong>Type:</strong> {res.type}</p>

                {res.photo && (
                  <img
                    src={`http://localhost:3001/${res.photo}`}
                    className="card-img-top"
                    alt={res.restaurantName}
                    style={{ height: "200px", objectFit: "cover", width: "100%", borderRadius: "8px" }}
                  />
                )}
                <button style={buttonStyle} onClick={() => handleDelete(res._id)}>
                  Delete Restaurant
                </button>
              </div>
            ))
          ) : (
            <p>No restaurants available.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Adminres;
