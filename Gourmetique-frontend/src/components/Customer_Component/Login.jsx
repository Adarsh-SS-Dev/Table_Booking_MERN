import React, { useState } from 'react';
import Nav from '../mainhome/Nav';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:3001/cuslogin', {
        username,
        password,
      });

      if (response.data.token) {
        sessionStorage.setItem('custoken', response.data.token);
        sessionStorage.setItem('customerid', response.data.customer.id);
        sessionStorage.setItem('customername', response.data.customer.name);
        sessionStorage.setItem('customerusername', response.data.customer.username);
        sessionStorage.setItem('customeraddress', response.data.customer.address);
        sessionStorage.setItem('customerpho', response.data.customer.mobileno);
        sessionStorage.setItem('customeremail', response.data.customer.email);
        sessionStorage.setItem('customergen', response.data.customer.gender);
        setError('');
        navigate('/hotellist');
      }
    } catch (err) {
      const errorMessage =
        err?.response?.data?.message ||
        'Login failed. Invalid Credentials. Please try again.';
      setError(errorMessage);
      console.log('Login error:', err);
    }
  };

  return (
    <div>
      <Nav />
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          backgroundColor: 'white',
        }}
      >
        <div
          style={{
            width: '500px',
            height: 'auto',
            backgroundColor: 'white',
            boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
            borderRadius: '10px',
            display: 'flex',
            flexDirection: 'column',
            padding: '30px',
          }}
        >
          <h1 className="text-center">
            <u>CUSTOMER LOGIN</u>
          </h1>

          {error && (
            <div className="alert alert-danger text-center mt-3">{error}</div>
          )}

          <form onSubmit={handleLogin}>
            <div
              className="mb-3"
              style={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                gap: '20px',
              }}
            >
              <label>Username</label>
              <input
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setError('');
                }}
                type="text"
                className="form-control"
                id="username"
                placeholder="Enter your username"
              />
            </div>

            <div
              className="mb-3"
              style={{
                display: 'flex',
                flexDirection: 'row',
                justifyContent: 'space-between',
                gap: '20px',
              }}
            >
              <label>Password</label>
              <input
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError('');
                }}
                type="password"
                className="form-control"
                id="password"
                placeholder="Enter your password"
              />
            </div>

            <div
              className="d-flex justify-content-center mb-3"
              style={{ gap: '20px' }}
            >
              <button className="btn btn-success" type="submit">
                Login
              </button>
            </div>
          </form>

          <hr
            style={{
              width: '80%',
              height: '2px',
              backgroundColor: 'black',
              border: 'none',
              margin: '10px auto',
            }}
          />

          
        </div>
      </div>
    </div>
  );
};

export default Login;
