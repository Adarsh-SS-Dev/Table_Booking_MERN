import React, { useEffect, useState } from 'react';
import 'bootstrap/dist/css/bootstrap.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import Nav from './Nav';
import axios from 'axios';

const Review = () => {

     const [review, setReview] = useState([]);
    const apiLink = "http://localhost:3001/viewreview";

    useEffect(() => {
        axios.get(apiLink)
          .then((response) => {
            setReview(response.data);
            console.log("Reviews:", response.data);
          });
        },[]);



    return (
        <div>
            <Nav />
            <div
                style={{
                    display: 'flex',
                    backgroundColor: 'rgb(179, 179, 179)',
                    justifyContent: 'center',
                    height: '100vh',
                    alignItems: 'center',
                    paddingTop: "20px"
                }}
            >
                <div className="review-container"
                    style={{
                        overflowY: 'auto',
                        height: "600px",
                        width: "500px",
                        display: 'flex',
                        flexDirection: 'column',  // Changed from row to column
                        backgroundColor: 'white',
                        alignItems: 'center',  // Center items horizontally
                        boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
                        borderRadius: '10px',
                        padding: '10px'
                    }}
                >
                    {
                        review.map((item, index) => {
                            return (
                                <div key={index} className="card mb-3" 
                                    style={{
                                        width: "90%",  // Limit width
                                        marginBottom: "10px"
                                    }}
                                >
                                    <img src={`http://localhost:3001/${item.image}`} className="card-img-top" alt="..." />
                                    <div className="card-body">
                                        <h5 className="card-title">{item.title}</h5>
                                        <p className="card-text"><strong>Hotel:</strong> {item.hotelName}</p>
                                    </div>
                                </div>
                            )
                        })
                    }
                </div>
            </div>
        </div>
    );
}

export default Review;
