import React, { useEffect, useState } from "react";
import Customer_Nav from "./Customer_Nav";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const Profile = () => {
  const navigate = useNavigate();
  const customerId = sessionStorage.getItem("customerid");
  const customerName = sessionStorage.getItem("customername");
  const customerUserName = sessionStorage.getItem("customerusername");
  const customerAdd = sessionStorage.getItem("customeraddress");
  const customerPhNo = sessionStorage.getItem("customerpho");
  const customerEmail = sessionStorage.getItem("customeremail");
  const customerGen = sessionStorage.getItem("customergen");

  const [order, setOrder] = useState([]);

  const api = "http://localhost:3001/getUserOrders";

  useEffect(() => {
    if (!customerId) {
      navigate('/');
    } else {
      axios.post(api, { customerId })
        .then((response) => {
          setOrder(response.data);
          console.log(response.data);
        })
        .catch((error) => console.log('Error fetching orders:', error));
    }
  }, [customerId, navigate]);

  const sidebarStyle = {
    width: "250px",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "100vh",
    backgroundColor: "white",
    padding: "10px 10px 10px 250px",
  };

  const cardWrapperStyle = {
    width: "1200px",
    height: "700px",
    backgroundColor: "white",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
    borderRadius: "10px",
    padding: "20px",
    display: "flex",
    flexDirection: "column",
  };

  const cardContainerStyle = {
    display: "flex",
    flexDirection: "column",
    overflowY: "auto",
    height: "100%",
    paddingTop: "10px",
  };

  return (
    <div style={{ display: "flex" }}>
      <div style={sidebarStyle}>
        <Customer_Nav />
      </div>

      <div style={containerStyle}>
        <div style={cardWrapperStyle}>
          <div style={cardContainerStyle}>
            <div className="card mb-3">
              <div className="card-header"><strong>{customerName ? customerName.toUpperCase() : ''}</strong></div>
              <div className="card-body">
                <blockquote className="blockquote mb-0">
                  <p>Username : {customerUserName}</p>
                  <p>Gender : {customerGen}</p>
                  <p>Address : {customerAdd}</p>
                  <p>Phone Number : {customerPhNo}</p>
                  <p>Email : {customerEmail}</p>
                </blockquote>
              </div>
            </div>

            <h2>Orders</h2>
            <div style={{ overflowX: "auto" }}>
              <table border={1} cellPadding={10} style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    <th>Order ID</th>
                    <th>Hotel Name</th>
                    <th>Order Date</th>
                    <th>Time Slot</th>
                    <th>Grand Total</th>
                    <th>No. of Customers</th>
                    <th>Items</th>
                  </tr>
                </thead>
                <tbody>
                  {order.length === 0 ? (
                    <tr>
                      <td colSpan={6}>No orders found.</td>
                    </tr>
                  ) : (
                    order.map((ord) => (
                      <tr key={ord._id}>
                        <td>{ord._id}</td>
                        <td>{ord.restaurantName}</td>
                        <td>{new Date(ord.orderDate).toLocaleDateString('en-GB')}</td>
                        <td>{ord.timeSlot}</td>
                        <td>₹{ord.grandTotal}</td>
                        <td>{ord.numberOfCustomers}</td>
                        <td>
                          {ord.items.map((item, idx) => (
                            <div key={idx}>
                              {item.name} × {item.count} ({item.category})
                            </div>
                          ))}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
