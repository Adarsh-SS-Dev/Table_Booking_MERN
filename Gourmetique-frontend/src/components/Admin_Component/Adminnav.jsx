import React from 'react'
import { Link, useNavigate } from 'react-router-dom'

const Adminnav = () => {
  const navigate = useNavigate();

  const handleLogout = () => {
    sessionStorage.clear(); // Or remove specific items like: sessionStorage.removeItem('restaurantname');
    navigate('/'); // Redirect to login/home page
  };

  return (
    <nav className="d-flex flex-column bg-dark text-white vh-100 p-3" style={{ width: '250px', position: 'fixed' }}>
      <div className="mb-4">
        <h4>Gourmetique</h4>
      </div>
      <ul className="nav flex-column">
        <li className="nav-item">
          <Link className="nav-link text-white" to="/admincus">Customer</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" to="/adminres">Restaurant</Link>
        </li>
        {/* <li className="nav-item">
        <Link className="nav-link text-white" to="/Weather">W</Link>
      </li> */}
        <li className="nav-item">
          <Link className="nav-link text-white" to="/review_ads">Ads & Reviews</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link text-white" onClick={handleLogout}>LogOut</Link>
        </li>
      </ul>
    </nav>
  )
}

export default Adminnav