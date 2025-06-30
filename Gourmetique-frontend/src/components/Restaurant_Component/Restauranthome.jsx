import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Restaurant_Nav from './Restaurant_Nav';
import { useNavigate } from 'react-router-dom';

const Restauranthome = () => {
  const [orders, setOrders] = useState([]);
  const [acceptedOrders, setAcceptedOrders] = useState([]);

  const hotelId = sessionStorage.getItem("restaurantid");
  const navigate = useNavigate();

  const convertTo24HourWithAMPM = (time) => {
    const [hour, minute] = time.split(':');
    const hourInt = parseInt(hour, 10);
    const suffix = hourInt >= 12 ? 'PM' : 'AM';
    const hour12 = hourInt % 12 || 12;
    return `${hour12}:${minute} ${suffix}`;
  };

  useEffect(() => {
    if (!hotelId) {
      navigate('/');
      return;
    }

    const fetchOrders = async () => {
      try {
        // Fetch accepted orders
        const acceptedOrdersResponse = await axios.post('http://localhost:3001/getAcceptedOrders', { hotelId });
        const acceptedOrdersList = acceptedOrdersResponse.data || [];

        const acceptedOrderIds = acceptedOrdersList.map(order => order.orderId);

        const formattedAcceptedOrders = acceptedOrdersList.map(order => {
          const date = new Date(order.orderDate);
          const formattedDate = date.toLocaleDateString('en-GB');
          const formattedTime = convertTo24HourWithAMPM(order.timeSlot);

          return {
            id: order.orderId,
            name: order.customerName,
            date: `${formattedDate} - ${formattedTime}`,
            total: `Rs.${order.grandTotal}`,
            items: order.items.map(item => ({
              name: item.name,
              quantity: item.count
            }))
          };
        });

        setAcceptedOrders(formattedAcceptedOrders);

        // Fetch all orders
        const response = await axios.post('http://localhost:3001/vieworders', { hotelId });
        const allOrders = response.data?.orders || [];

        const filteredNewOrders = allOrders
          .filter(order => !acceptedOrderIds.includes(order._id))
          .map(order => {
            const date = new Date(order.orderDate);
            const formattedDate = date.toLocaleDateString('en-GB');
            const formattedTime = convertTo24HourWithAMPM(order.timeSlot);

            return {
              id: order._id,
              name: order.customerName,
              date: `${formattedDate} - ${formattedTime}`,
              total: `Rs.${order.grandTotal}`,
              items: order.items.map(item => ({
                name: item.name,
                quantity: item.count
              })),
              timeSlot: order.timeSlot
            };
          });

        setOrders(filteredNewOrders);

      } catch (error) {
        console.error('Failed to fetch orders:', error);
      }
    };

    fetchOrders();
  }, [hotelId, navigate]);

  const handleAccept = async (orderId) => {
    const acceptedOrder = orders.find((order) => order.id === orderId);
    if (acceptedOrder) {
      setAcceptedOrders((prev) => [...prev, acceptedOrder]);
      setOrders((prev) => prev.filter((order) => order.id !== orderId));

      try {
        await axios.post('http://localhost:3001/saveAcceptedOrder', {
          orderId: acceptedOrder.id,
          customerName: acceptedOrder.name,
          orderDate: acceptedOrder.date.split(' - ')[0],
          grandTotal: acceptedOrder.total.replace('Rs.', ''),
          items: acceptedOrder.items.map(item => ({
            name: item.name,
            count: item.quantity,
          })),
          hotelId: hotelId,
          timeSlot: acceptedOrder.timeSlot,
        });

      } catch (error) {
        console.error('Failed to save accepted order:', error);
      }
    }
  };

  const sidebarStyle = {
    width: "250px",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  };

  const containerStyle = {
    display: "flex",
    justifyContent: "center",
    alignItems: "flex-start",
    height: "100vh",
    backgroundColor: "white",
    padding: "20px 20px 20px 250px",
    overflowY: "auto",
  };

  const contentStyle = {
    width: "100%",
    maxWidth: "1000px",
    padding: "20px",
    borderRadius: "10px",
    backgroundColor: "white",
    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
  };

  const cardContainerStyle = {
    display: "flex",
    flexWrap: "wrap",
    gap: "20px",
  };

  const cardStyle = {
    flex: "1 1 calc(50% - 20px)",
    backgroundColor: "#f8f9fa",
    padding: "15px",
    borderRadius: "8px",
    boxShadow: "0px 2px 6px rgba(0,0,0,0.1)",
  };

  return (
    <div style={{ display: 'flex' }}>
      <div style={sidebarStyle}>
        <Restaurant_Nav />
      </div>

      <div style={containerStyle}>
        <div style={contentStyle}>
          <h3 style={{ marginTop: "40px" }}>New Orders</h3>
          <div style={cardContainerStyle}>
            {orders.length > 0 ? (
              orders.map((order) => (
                <div key={order.id} style={cardStyle}>
                  <h4>Order Id: {order.id}</h4>
                  <p><strong>Customer:</strong> {order.name}</p>
                  <p><strong>Date:</strong> {order.date}</p>
                  <p><strong>Total:</strong> {order.total}</p>
                  <p><strong>Items:</strong></p>
                  <ul>
                    {order.items.map((item, index) => (
                      <li key={index}>{item.name} x {item.quantity}</li>
                    ))}
                  </ul>
                  <button
                    onClick={() => handleAccept(order.id)}
                    style={{ backgroundColor: "green", color: "white", padding: "8px 12px", border: "none", borderRadius: "4px", marginTop: "10px" }}
                  >
                    Accept
                  </button>
                </div>
              ))
            ) : (
              <div style={{
                width: "100%",
                textAlign: "center",
                padding: "40px",
                fontSize: "18px"
              }}>
                No new orders
              </div>
            )}
          </div>
          <h3 style={{ marginTop: "40px" }}>Accepted Orders</h3>
          {acceptedOrders.length > 0 ? (
            <div style={cardContainerStyle}>
              {acceptedOrders.map((order) => (
                <div key={order.id} style={cardStyle}>
                  <h4>Order Id: {order.id}</h4>
                  <p><strong>Customer:</strong> {order.name}</p>
                  <p><strong>Date:</strong> {order.date}</p>
                  <p><strong>Total:</strong> {order.total}</p>
                  <p><strong>Items:</strong></p>
                  <ul>
                    {order.items.map((item, index) => (
                      <li key={index}>{item.name} x {item.quantity}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ) : (
            <p>No accepted orders</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Restauranthome;
