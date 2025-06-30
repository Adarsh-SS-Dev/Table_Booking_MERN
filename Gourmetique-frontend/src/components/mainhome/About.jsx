import React from 'react'
import Nav from './Nav'

const About = () => {
  return (
    <div>
      <Nav/>
    <div
    style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      height: "100vh", // Full viewport height
      backgroundColor: "white",
    }}
  >
    <div
      style={{
        width: "500px", // Adjust as needed
        height: "500px", // Adjust as needed
        backgroundColor: "white",
        boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)", // Optional shadow
        borderRadius: "10px",
        display: "flex",
        flexDirection:"column"
        
      }}
    >
    <h3><u>GOURMETIQUE</u></h3><br />
    <p>Our mission is to provide the best quality food and drinks to our customers.</p>
    </div>
  </div>
  </div>

  )
}

export default About