import React, { useState } from "react";
import Nav from "../mainhome/Nav";
import axios from "axios";

const Resregistration = () => {
  const [inputField, setInputField] = useState({
    restaurantName: "",
    ownerName: "",
    email: "",
    phone: "",
    address: "",
    type: "",
    username: "",
    password: "",
    photo: "",
  });

  const [file, setFile] = useState(null);

  const apiLink = "http://localhost:3001/restaurantregistration";

  const inputHandler = (event) => {
    setInputField({ ...inputField, [event.target.name]: event.target.value });
  };

  const fileHandler = (event) => {
    setFile(event.target.files[0]); // Store the file separately
  };

  const readValue = () => {
    const formData = new FormData();

    // Append all input fields to FormData
    Object.keys(inputField).forEach((key) => {
      formData.append(key, inputField[key]);
    });

    // Append file separately
    if (file) {
      formData.append("photo", file);
    }

    axios
      .post(apiLink, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        console.log(response.data);
        if (response.data.message === "Restaurant registered successfully") {
          alert("Restaurant registered successfully");
          setInputField({
            restaurantName: "",
            ownerName: "",
            email: "",
            phone: "",
            address: "",
            type: "",
            username: "",
            password: "",
            photo: "",
          });
          setFile(null);
        } else {
          alert("Registration failed!!");
        }
      })
      .catch((error) => {
        console.error(error);
        alert("An error occurred while registering");
      });
  };

  return (
    <div>
      <Nav />
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "1000px",
        }}
      >
        <div
          style={{
            width: "850px",
            height: "750px",
            backgroundColor: "white",
            boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
            borderRadius: "10px",
            display: "flex",
            flexDirection: "column",
            padding: "20px",
          }}
        >
          <h1
            style={{
              paddingBottom: "20px",
            }}
          >
            <u>RESTAURANT REGISTRATION</u>
          </h1>
          <br />

          {/* Row 1: Restaurant Name & Owner Name */}
          <div
            style={{
              display: "flex",
              alignItems: "center", // Align items properly
              justifyContent: "space-between",
              gap: "20px",
              paddingBottom: "20px",
            }}
          >
            {/* Restaurant Name */}
            <div style={{ width: "48%" }}>
              <label htmlFor="restaurantName">Enter Restaurant Name</label>
              <input
                onChange={inputHandler}
                type="text"
                className="form-control"
                id="restaurantName"
                placeholder="Restaurant Name"
                name="restaurantName"
                value={inputField.restaurantName}
              />
            </div>

            {/* Vertical Line */}
            <div
              style={{
                width: "2px",
                height: "50px", // Adjust height as needed
                backgroundColor: "#ccc",
              }}
            ></div>

            {/* Owner's Name */}
            <div style={{ width: "48%" }}>
              <label htmlFor="ownerName">Enter Owner's Name</label>
              <input
                onChange={inputHandler}
                type="text"
                className="form-control"
                id="ownerName"
                placeholder="Owner's Name"
                name="ownerName"
                value={inputField.ownerName}
              />
            </div>
          </div>

          {/* Row 2: Email Address & Phone Number */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              paddingBottom: "20px",
            }}
          >
            {/* Email Address */}
            <div style={{ width: "48%" }}>
              <label htmlFor="email">Enter Email Address</label>
              <input
                onChange={inputHandler}
                type="email"
                className="form-control"
                id="email"
                placeholder="Email Address"
                name="email"
                value={inputField.email}
              />
            </div>

            {/* Vertical Line */}
            <div
              style={{
                width: "2px",
                height: "50px",
                backgroundColor: "#ccc",
              }}
            ></div>

            {/* Phone Number */}
            <div style={{ width: "48%" }}>
              <label htmlFor="phone">Enter Phone Number</label>
              <input
                onChange={inputHandler}
                type="text"
                className="form-control"
                id="phone"
                placeholder="Phone Number"
                name="phone"
                value={inputField.phone}
              />
            </div>
          </div>

          {/* Row 3: Address & Type */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              paddingBottom: "20px",
            }}
          >
            {/* Address */}
            <div style={{ width: "48%" }}>
              <label htmlFor="address">Enter Address</label>
              <input
                onChange={inputHandler}
                type="text"
                className="form-control"
                id="address"
                placeholder="Address"
                name="address"
                value={inputField.address}
              />
            </div>

            {/* Vertical Line */}
            <div
              style={{
                width: "2px",
                height: "50px",
                backgroundColor: "#ccc",
              }}
            ></div>

            {/* type */}
            <div style={{ width: "48%" }}>
              <label htmlFor="type">Veg/Non-Veg</label>
              <select
                onChange={inputHandler}
                name="type"
                className="form-control"
                id="type"
                value={inputField.type}
              >
                <option value="">Select Type</option>
                <option value="veg">Veg</option>
                <option value="non-veg">Non-Veg</option>
                <option value="both">Both</option>
              </select>
            </div>
          </div>

          {/* Row 4: photo upload & Username */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              paddingBottom: "20px",
            }}
          >
            {/* photo */}
            <div style={{ width: "48%" }}>
              <label htmlFor="photo">Photo of Restaurant</label>
              <input
                onChange={fileHandler}
                accept="image/*"
                type="file"
                className="form-control"
                id="photo"
                name="photo"
              />
            </div>

            {/* Vertical Line */}
            <div
              style={{
                width: "2px",
                height: "50px",
                backgroundColor: "#ccc",
              }}
            ></div>

            {/* Username */}
            <div style={{ width: "48%" }}>
              <label htmlFor="username">Enter Username</label>
              <input
                onChange={inputHandler}
                type="text"
                className="form-control"
                id="username"
                placeholder="Username"
                name="username"
                value={inputField.value}
              />
            </div>
          </div>

          {/* Row 5: Password & reenterpassword */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              paddingBottom: "20px",
            }}
          >
            {/* password */}
            <div style={{ width: "48%" }}>
              <label htmlFor="password">Enter The Password</label>
              <input
                onChange={inputHandler}
                type="password"
                className="form-control"
                id="password"
                placeholder="Password"
                name="password"
                value={inputField.password}
              />
            </div>

            {/* Vertical Line */}
            <div
              style={{
                width: "2px",
                height: "50px",
                backgroundColor: "#ccc",
              }}
            ></div>

            {/* reenter password */}
            <div style={{ width: "48%" }}>
              <label htmlFor="repassword">Re-Enter Password</label>
              <input
                type="password"
                className="form-control"
                id="reenpassword"
                placeholder="Re-enter password"
              />
            </div>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "20px",
              paddingTop: "30px",
            }}
          >
            {/* button */}
            <div style={{ width: "100%" }}>
              <button
                onClick={readValue}
                className="btn btn-success"
                type="submit"
              >
                Register
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Resregistration;
