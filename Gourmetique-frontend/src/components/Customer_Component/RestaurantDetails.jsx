import React, { useEffect, useState, useRef } from "react";
import Customer_Nav from "./Customer_Nav";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const RestaurantDetails = () => {
  const [hotelData, setHotelData] = useState({});
  const [selectedItems, setSelectedItems] = useState([]);
  const [itemStates, setItemStates] = useState({});
  const [numCustomers, setNumCustomers] = useState("");
  const [timeSlot, setTimeSlot] = useState("");
  const [menu, setMenu] = useState({
    starters: [],
    mainCourse: [],
    desserts: [],
  });
  const [showInvoice, setShowInvoice] = useState(false);
  const [predictedCustomers, setPredictedCustomers] = useState(null);
  const [loading, setLoading] = useState(false); // Loader state for API requests
  const invoiceRef = useRef();
  const navigate = useNavigate();
  const customerId = sessionStorage.getItem("customerid");
  const hotelid = sessionStorage.getItem("hotelId");
  const apiUrl = "http://localhost:3001/viewsprestaurant";

  useEffect(() => {
    if (!customerId) {
      navigate("/");
    }

    // Fetch hotel data
    axios
      .post(apiUrl, { _id: hotelid })
      .then((response) => setHotelData(response.data))
      .catch((error) => console.error("Error fetching restaurant:", error));

    // Fetch menu items
    const fetchMenuItems = async () => {
      setLoading(true); // Start loading
      try {
        const [starterRes, mainRes, dessertRes] = await Promise.all([
          axios.post("http://localhost:3001/viewstarters", { hotelid }),
          axios.post("http://localhost:3001/viewmains", { hotelid }),
          axios.post("http://localhost:3001/viewdesserts", { hotelid }),
        ]);
        setMenu({
          starters: starterRes.data.map((item) => ({
            id: item._id,
            name: item.item,
            price: item.price,
          })),
          mainCourse: mainRes.data.map((item) => ({
            id: item._id,
            name: item.mitem,
            price: item.mprice,
          })),
          desserts: dessertRes.data.map((item) => ({
            id: item._id,
            name: item.ditem,
            price: item.dprice,
          })),
        });
      } catch (error) {
        console.error("Error fetching menu items:", error);
      } finally {
        setLoading(false); // Stop loading
      }
    };

    fetchMenuItems();

    // Fetch busy time prediction from FastAPI
    const fetchBusyTimePrediction = async () => {
      try {
        const currentTime = new Date();
        const hours = String(currentTime.getHours()).padStart(2, "0");
        const minutes = String(currentTime.getMinutes()).padStart(2, "0");
        const time = `${hours}:${minutes}`;
        const response = await axios.post("http://localhost:8000/predict", {
          hotelId: hotelid,
          timeSlot: time,
        });
        setPredictedCustomers(response.data.predictedNumberOfCustomers);
      } catch (error) {
        console.error("Error fetching busy time prediction:", error);
      }
    };

    fetchBusyTimePrediction();
  }, [customerId, navigate, hotelid]);

  const handleSelectionChange = (category, item, checked, count) => {
    const id = `${category}_${item.id}`;
    setItemStates((prev) => ({
      ...prev,
      [id]: { checked, count: count || 1 },
    }));

    const newItems = [...selectedItems];
    const index = newItems.findIndex(
      (i) => i.id === item.id && i.category === category
    );

    if (checked) {
      const newItem = {
        id: item.id,
        name: item.name,
        category,
        price: item.price,
        count: count || 1,
      };
      if (index === -1) newItems.push(newItem);
      else newItems[index].count = count || 1;
    } else {
      if (index !== -1) newItems.splice(index, 1);
    }

    setSelectedItems(newItems);
  };

  const handleCountChange = (category, item, count) => {
    const id = `${category}_${item.id}`;
    setItemStates((prev) => ({
      ...prev,
      [id]: { ...prev[id], count },
    }));

    const newItems = [...selectedItems];
    const index = newItems.findIndex(
      (i) => i.id === item.id && i.category === category
    );
    if (index !== -1) {
      newItems[index].count = count;
      setSelectedItems(newItems);
    }
  };

  const calculateGrandTotal = () => {
    return selectedItems.reduce(
      (total, item) => total + item.price * item.count,
      0
    );
  };

  const loadRazorpayScript = () => {
    return new Promise((resolve) => {
      const script = document.createElement("script");
      script.src = "https://checkout.razorpay.com/v1/checkout.js";
      script.onload = () => resolve(true);
      script.onerror = () => resolve(false);
      document.body.appendChild(script);
    });
  };

  const handleSubmit = () => {
    if (selectedItems.length === 0) {
      alert("Please select at least one item to place the order.");
      return;
    }

    if (!numCustomers || !timeSlot) {
      alert("Please fill in the number of customers and time slot.");
      return;
    }

    setShowInvoice(true);
  };

  const proceedToPayment = async () => {
    setShowInvoice(false);

    const res = await loadRazorpayScript();
    if (!res) {
      alert("Failed to load Razorpay SDK.");
      return;
    }

    const grandTotal = calculateGrandTotal();
    const customerId = sessionStorage.getItem("customerid");

    const currentDate = new Date().toISOString().split("T")[0];

    const options = {
      key: "rzp_test_03rYoJSFvAmjfB",
      amount: grandTotal * 100,
      currency: "INR",
      name: "Gourmetique",
      description: "Food Order Payment",
      handler: async function (response) {
        const paymentId = response.razorpay_payment_id;

        const orderData = {
          hotelId: hotelid,
          items: selectedItems,
          numberOfCustomers: numCustomers,
          timeSlot: timeSlot,
          grandTotal: grandTotal,
          customerId: customerId,
          paymentId: paymentId,
          orderDate: currentDate,
        };

        try {
          await axios.post("http://localhost:3001/placeorder", orderData);
          alert("Order placed successfully!");
          setSelectedItems([]);
          setItemStates({});
          setNumCustomers("");
          setTimeSlot("");
        } catch (err) {
          console.error("Error placing order:", err);
          alert("Failed to place order after payment.");
        }
      },
      prefill: {
        name: "Customer",
        email: "customer@example.com",
        contact: "9999999999",
      },
      theme: {
        color: "#3399cc",
      },
    };

    const rzp = new window.Razorpay(options);
    rzp.open();
  };

  const renderTable = (category, items) => (
    <div>
      <h2>{category}</h2>
      <table border={2} cellPadding={10}>
        <thead>
          <tr>
            <th>SI</th>
            <th>ITEM</th>
            <th>PRICE</th>
            <th>COUNT</th>
            <th>SELECTED</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => {
            const id = `${category}_${item.id}`;
            const state = itemStates[id] || { checked: false, count: 1 };

            return (
              <tr key={item.id}>
                <td>{idx + 1}</td>
                <td>{item.name}</td>
                <td>Rs.{item.price}</td>
                <td>
                  <input
                    type="number"
                    min={1}
                    value={state.count}
                    onChange={(e) =>
                      handleCountChange(
                        category,
                        item,
                        parseInt(e.target.value)
                      )
                    }
                  />
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={state.checked}
                    onChange={(e) =>
                      handleSelectionChange(
                        category,
                        item,
                        e.target.checked,
                        state.count
                      )
                    }
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  const handlePrint = () => {
    const printContents = invoiceRef.current.innerHTML;
    const win = window.open("", "", "height=600,width=800");
    win.document.write("<html><head><title>Invoice</title></head><body>");
    win.document.write(printContents);
    win.document.write("</body></html>");
    win.document.close();
    win.print();
  };

  const sidebarStyle = {
    width: "250px",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  };

  const mainContainerStyle = {
    flexGrow: 1,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "20px",
    backgroundColor: "#f9f9f9",
    height: "100vh",
    overflow: "hidden",
  };

  const cardStyle = {
    height: "100%",
    width: "700px",
    backgroundColor: "white",
    padding: "20px",
    borderRadius: "8px",
    overflowY: "auto",
    boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
  };

  const inputGroupStyle = {
    marginTop: "20px",
    padding: "15px",
    border: "1px solid #ddd",
    borderRadius: "8px",
    backgroundColor: "#f5f5f5",
  };

  return (
    <div style={{ display: "flex" }}>
      <div style={sidebarStyle}>
        <Customer_Nav />
      </div>

      <div style={mainContainerStyle}>
        <div style={cardStyle}>
          <div className="card mb-3">
            <div
              className="card-header"
              style={{ fontSize: "24px", fontWeight: "bold" }}
            >
              {hotelData.restaurantName}
            </div>
            <div className="card-body">
              <p className="card-text">Address: {hotelData.address}</p>
              <p className="card-text">Phone No. : {hotelData.phone}</p>
              <h3>Busy Time Of The Restaurant:</h3>
              {predictedCustomers !== null ? (
                <div style={{ fontSize: "18px", fontWeight: "bold" }}>
                  <span
                    style={{
                      color:
                        predictedCustomers <= 10
                          ? "green"
                          : predictedCustomers <= 20
                          ? "orange"
                          : "red",
                    }}
                  >
                    Predicted Customers: {predictedCustomers}
                  </span>
                  {predictedCustomers > 20 && (
                    <div
                      style={{
                        color: "red",
                        fontWeight: "bold",
                        marginTop: "5px",
                      }}
                    >
                      ⚠️ High expected crowd. You may experience delays.
                    </div>
                  )}
                </div>
              ) : (
                <p>Loading prediction...</p>
              )}
            </div>
          </div>

          {loading ? (
            <p>Loading menu items...</p>
          ) : (
            <>
              {renderTable("Starters", menu.starters)}
              {renderTable("Main Course", menu.mainCourse)}
              {renderTable("Desserts", menu.desserts)}
            </>
          )}

          <textarea
            style={{ width: "100%", marginTop: "10px", height: "100px" }}
            readOnly
            value={selectedItems
              .map(
                (item) =>
                  `${item.name} (x${item.count}) - Rs.${
                    item.price * item.count
                  }`
              )
              .join("\n")}
          ></textarea>

          <div style={{ marginTop: "10px", fontWeight: "bold" }}>
            Grand Total: Rs.{calculateGrandTotal()}
          </div>

          <div style={inputGroupStyle}>
            <div className="mb-2">
              <label>Number of Customers</label>
              <input
                type="number"
                min={1}
                value={numCustomers}
                onChange={(e) => setNumCustomers(e.target.value)}
                className="form-control"
                placeholder="Enter number of people"
              />
            </div>
            <div className="mb-2">
              <label>Time Slot</label>
              <input
                type="time"
                value={timeSlot}
                onChange={(e) => setTimeSlot(e.target.value)}
                className="form-control"
              />
            </div>
          </div>

          <button
            type="button"
            onClick={handleSubmit}
            style={{ marginTop: "10px", width: "100%", padding: "10px" }}
          >
            Place Order
          </button>
        </div>
      </div>

      {showInvoice && (
        <div
          style={{
            position: "fixed",
            top: "0",
            left: "0",
            width: "100vw",
            height: "100vh",
            background: "rgba(0,0,0,0.6)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 1000,
          }}
        >
          <div
            style={{
              width: "400px",
              background: "#fff",
              padding: "20px",
              borderRadius: "10px",
              boxShadow: "0 4px 10px rgba(0,0,0,0.3)",
            }}
            ref={invoiceRef}
          >
            <h2 style={{ textAlign: "center", color: "#3399cc" }}>
              Gourmetique
            </h2>
            <h3 style={{ textAlign: "center" }}>Invoice</h3>
            <hr />
            <ul>
              {selectedItems.map((item, index) => (
                <li key={index}>
                  {item.name} (x{item.count}) — Rs.{item.price * item.count}
                </li>
              ))}
            </ul>
            <p>
              <strong>Number of Customers:</strong> {numCustomers}
            </p>
            <p>
              <strong>Time Slot:</strong> {timeSlot}
            </p>
            <h4>Total: Rs.{calculateGrandTotal()}</h4>
            <div
              style={{
                marginTop: "20px",
                display: "flex",
                justifyContent: "center",
                gap: "10px",
              }}
            >
              <button onClick={handlePrint}>Print</button>
              <button onClick={proceedToPayment}>Proceed to Pay</button>
              <button onClick={() => setShowInvoice(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RestaurantDetails;
