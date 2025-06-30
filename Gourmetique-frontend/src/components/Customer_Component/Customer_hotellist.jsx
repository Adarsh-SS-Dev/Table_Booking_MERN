import React, { useState, useEffect } from "react";
import axios from "axios";
import Customer_Nav from "./Customer_Nav";
import { useNavigate } from "react-router-dom";

const Customer_hotellist = () => {
  const [hotelData, setHotelData] = useState([]);
  const [filteredHotels, setFilteredHotels] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");

  const apiUrl = "http://localhost:3001/viewrestaurant";
  const navigate = useNavigate(); // For navigation
  const customerId = sessionStorage.getItem("customerid");

  useEffect(() => {
    if (!customerId) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
    axios
      .get(apiUrl)
      .then((response) => {
        setHotelData(response.data);
        setFilteredHotels(response.data);
      })
      .catch((error) => console.error("Error fetching restaurants:", error));
  }, [customerId, navigate]);

  const handleSearchChange = (event) => {
    const query = event.target.value.toLowerCase();
    setSearchQuery(query);

    const filtered = hotelData.filter((hotel) =>
      hotel.restaurantName.toLowerCase().includes(query)
    );
    setFilteredHotels(filtered);
  };

  const handleCardClick = (id) => {
    // Store the hotel ID in sessionStorage
    sessionStorage.setItem("hotelId", id);
    navigate("/restaurant");
  };

  // Styles (same as before)
  const sidebarStyle = {
    width: "250px",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "white",
    padding: "10px 10px 10px 250px",
  };

  const cardWrapperStyle = {
    width: "800px",
    height: "500px",
    backgroundColor: "white",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
    borderRadius: "10px",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
  };

  const searchBarStyle = {
    position: "sticky",
    top: 0,
    backgroundColor: "white",
    paddingBottom: "10px",
    zIndex: 10,
  };

  const cardContainerStyle = {
    display: "flex",
    flexWrap: "wrap",
    justifyContent: "space-between",
    gap: "20px",
    overflowY: "auto",
    height: "100%",
    paddingTop: "10px",
  };

  const cardStyle = {
    width: "18rem",
    flex: "1 1 calc(33.33% - 20px)",
    maxWidth: "18rem",
    cursor: "pointer", // Make it look clickable
  };

  return (
    <div style={{ display: "flex" }}>
      <div style={sidebarStyle}>
        <Customer_Nav />
      </div>

      <div style={containerStyle}>
        <div style={cardWrapperStyle}>
          <div style={searchBarStyle}>
            <form className="d-flex" role="search">
              <input
                className="form-control me-2"
                type="search"
                placeholder="Search"
                aria-label="Search"
                value={searchQuery}
                onChange={handleSearchChange}
              />
              <button className="btn btn-outline-success" type="button">
                Search
              </button>
            </form>
          </div>

          <div style={cardContainerStyle}>
            {filteredHotels.length > 0 ? (
              filteredHotels.map((hotel) => (
                <div
                  key={hotel._id}
                  className="card"
                  style={cardStyle}
                  onClick={() => handleCardClick(hotel._id)}
                >
                  <img
                    src={`http://localhost:3001/${hotel.photo}`}
                    className="card-img-top"
                    alt={hotel.restaurantName}
                    style={{ height: "200px", objectFit: "cover" }}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{hotel.restaurantName}</h5>
                    <p className="card-text">{hotel.address}</p>
                    <p className="card-text">
                      Type: <strong>{hotel.type}</strong>
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <p>No restaurants found.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Customer_hotellist;
