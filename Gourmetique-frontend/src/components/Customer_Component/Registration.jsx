import React, { useState } from 'react';
import Nav from '../mainhome/Nav';
import axios from 'axios';

const Registration = () => {
  const [inputField,setInputField] =useState({
    name:"",
    mobileno:"",
    email:"",
    address:"",
    gender:"",
    username:"",
    password:""
  });
  const apiLink ="http://localhost:3001/cusreg";

  const inputHandler = (event) => {
    setInputField({ ...inputField, [event.target.name]: event.target.value });
  };

  const readValue = () => {
    axios.post(apiLink, inputField, {
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        console.log(response.data);
        if (response.data.message === "Customer registered successfully") {
            alert("Customer registered successfully");
            setInputField({
                name: "",
                mobileno: "",
                email: "",
                address: "",
                gender: "",
                username: "",
                password: ""
            });
        } else {
            alert("Customer registration failed");
        }
    })
    .catch(error => {
        console.error("Axios Error:", error);
        alert("An error occurred while registering the customer.");
    });
  };
  
    


  return (
    <div>
      <Nav />

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '100vh',
          backgroundColor: '#f8f9fa',
          padding: '20px',
        }}
      >
        <div
          style={{
            width: '850px',
            backgroundColor: 'white',
            boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
            borderRadius: '10px',
            padding: '20px',
          }}
        >
          <h1 style={{ paddingBottom: '20px', textAlign: 'center' }}>
            <u>CUSTOMER REGISTRATION</u>
          </h1>

          {/* Row 1: Customer Name & Mobile */}
          <div className="d-flex justify-content-between pb-3">
            <div style={{ width: '48%' }}>
              <label htmlFor="name">Enter Customer Name</label>
              <input onChange={inputHandler}
              type="text"
               className="form-control"
                id="name"
                 placeholder="Customer Name"
              name='name' 
              value={inputField.name}/>
            </div>
            <div style={{ width: '2px', backgroundColor: '#ccc' }}></div>
            <div style={{ width: '48%' }}>
              <label htmlFor="mobileno">Enter Mobile Number</label>
              <input 
               onChange={inputHandler}
               name='mobileno'
               value={inputField.mobileno} type="text" className="form-control" id="mobileno" placeholder="Mobile Number" />
            </div>
          </div>

          {/* Row 2: Email & Address */}
          <div className="d-flex justify-content-between pb-3">
            <div style={{ width: '48%' }}>
              <label htmlFor="email">Enter E-mail Address</label>
              <input
              onChange={inputHandler}
              name='email' 
               value={inputField.email} type="email" className="form-control" id="email" placeholder="E-mail" />
            </div>
            <div style={{ width: '2px', backgroundColor: '#ccc' }}></div>
            <div style={{ width: '48%' }}>
              <label htmlFor="address">Enter Address</label>
              <input
              onChange={inputHandler}
              value={inputField.address}
              name='address' type="text" className="form-control" id="address" placeholder="Address" />
            </div>
          </div>

          {/* Row 3: Gender & Username */}
          <div className="d-flex justify-content-between pb-3">
            <div style={{ width: '48%' }}>
              <label htmlFor="gender">Select Gender</label>
              <select className="form-control" id="gender"
              name='gender' value={inputField.gender} onChange={inputHandler}>
                <option value="">Select Gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div style={{ width: '2px', backgroundColor: '#ccc' }}></div>
            <div style={{ width: '48%' }}>
              <label htmlFor="username">Enter Username</label>
              <input onChange={inputHandler}
              name='username'
              value={inputField.value}
               type="text" className="form-control" id="username" placeholder="Username" />
            </div>
          </div>

          {/* Row 4: Password & Re-enter Password */}
          <div className="d-flex justify-content-between pb-3">
            <div style={{ width: '48%' }}>
              <label htmlFor="password">Enter Password</label>
              <input 
              onChange={inputHandler}
              name='password'
              value={inputField.password} type="password" className="form-control" id="password" placeholder="Password" />
            </div>
            <div style={{ width: '2px', backgroundColor: '#ccc' }}></div>
            <div style={{ width: '48%' }}>
              <label htmlFor="repassword">Re-Enter Password</label>
              <input type="password" className="form-control" id="repassword" placeholder="Re-Enter Password" />
            </div>
          </div>

          {/* Submit Button */}
          <div className="text-center pt-3">
            <button 
            onClick={readValue}
            type='submit' className="btn btn-success w-50">Register</button>
          </div>
        </div>
      </div>
    </div>
  );
};


export default Registration;
