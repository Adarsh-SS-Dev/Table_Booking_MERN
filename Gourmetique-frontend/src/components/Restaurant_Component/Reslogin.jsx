import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import Nav from "../mainhome/Nav";

const Reslogin = () => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:3001/login", {
        username,
        password,
      });

      if (response.data.token) {
        sessionStorage.setItem("restoken", response.data.token);
        sessionStorage.setItem("restaurantid", response.data.restaurant.id); 
        sessionStorage.setItem("restaurantname", response.data.restaurant.restaurantName); 
        sessionStorage.setItem("restaurantownername", response.data.restaurant.ownerName); 
        sessionStorage.setItem("restaurantadd", response.data.restaurant.address); 
        sessionStorage.setItem("restaurantphNo", response.data.restaurant.phone);  
        setError('');
        navigate("/menu");
        console.log(response.data);
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
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          backgroundColor: "white",
        }}
      >
        <div
          style={{
            width: "500px",
            height: "500px",
            backgroundColor: "white",
            boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
            borderRadius: "10px",
            display: "flex",
            flexDirection: "column",
            padding: "20px",
          }}
        >
          <h1>
            <u>RESTAURANT LOGIN</u>
          </h1>

          {error && <p style={{ color: "red", textAlign: "center" }}>{error}</p>}

          <form onSubmit={handleLogin}>
            <div style={{ marginBottom: "15px" }}>
              <label>Username</label>
              <input
                type="text"
                className="form-control"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                required
              />
            </div>

            <div style={{ marginBottom: "15px" }}>
              <label>Password</label>
              <input
                type="password"
                className="form-control"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
              />
            </div>

            <button className="btn btn-success" type="submit">
              Login
            </button>
          </form>

          <br />
          <hr style={{ width: "80%", height: "2px", backgroundColor: "black", border: "none", margin: "10px auto" }} />

          {/* <button className="btn btn-success">Sign In With Google</button> */}
        </div>
      </div>
    </div>
  );
};

export default Reslogin;
