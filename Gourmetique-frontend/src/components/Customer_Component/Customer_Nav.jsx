import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import { Link, useNavigate } from 'react-router-dom';

const Customer_Nav = () => {
  const [customerName, setCustomerName] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const name = sessionStorage.getItem('customername');
    if (name) {
      setCustomerName(name);
    }
  }, []);

  const handleLogout = () => {
    sessionStorage.clear(); // Or remove specific items like: sessionStorage.removeItem('restaurantname');
    navigate('/'); // Redirect to login/home page
  };

  return (
    <nav className="d-flex flex-column bg-dark text-white vh-100 p-3" style={{ width: '250px', position: 'fixed' }}>
      <div className="mb-4">
        <h4>Gourmetique</h4>
        <h5>HELLO {customerName ? customerName.toUpperCase() : ''}</h5>
      </div>
      <ul className="nav flex-column">
        <li className="nav-item">
          <Link className="nav-link text-white" to="/hotellist">Restaurants</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/customerreview">Review</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/Weather">Weather</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/profile">Profile</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" onClick={handleLogout}>LogOut</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Customer_Nav;
