import React, { useEffect, useState } from 'react';
import Adminnav from './Adminnav';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Admincus = () => {
  const [customers, setCustomers] = useState([]);

  const apiUrl = "http://localhost:3001/viewcus";
  const apiUrl1 = "http://localhost:3001/delcus";
  const admtoken = sessionStorage.getItem("adtoken");
  const navigate = useNavigate();

  useEffect(() => {
    if (!admtoken) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
    axios
      .get(apiUrl)
      .then((response) => {
        setCustomers(response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  }, [admtoken, navigate]);

  const handleDelete = async (_id) => {
    try {
      const response = await axios.post(apiUrl1, { _id });
      if (response.status === 200) {
        setCustomers((prev) => prev.filter((cust) => cust._id !== _id));
        alert("Customer deleted successfully.");
      } else {
        alert("Failed to delete customer.");
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert("Error deleting customer.");
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
          <h3>Customer Details</h3>
          {customers.length > 0 ? (
            customers.map((customer) => (
              <div key={customer._id} style={cardStyle}>
                <p><strong>Name:</strong> {customer.name}</p>
                <p><strong>Email:</strong> {customer.email}</p>
                <p><strong>Phone:</strong> {customer.mobileno}</p>
                <p><strong>Address:</strong> {customer.address}</p>
                <p><strong>Gender:</strong> {customer.gender}</p>
                <button style={buttonStyle} onClick={() => handleDelete(customer._id)}>
                  Delete Customer
                </button>
              </div>
            ))
          ) : (
            <p>No customers available.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Admincus;
