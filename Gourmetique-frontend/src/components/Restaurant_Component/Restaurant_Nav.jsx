import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Restaurant_Nav = () => {
  const [restaurantName, setRestaurantName] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const name = sessionStorage.getItem('restaurantname');
    if (name) {
      setRestaurantName(name);
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
        <h5>HELLO {restaurantName ? restaurantName.toUpperCase() : ''}</h5>
      </div>
      <ul className="nav flex-column">
        <li className="nav-item">
          <Link className="nav-link text-white" to="/resthome">Orders</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/menu">Menu</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/resads">Post New Ads</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/resprofile">Profile</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white btn btn-link p-0" onClick={handleLogout}>LogOut</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Restaurant_Nav;
