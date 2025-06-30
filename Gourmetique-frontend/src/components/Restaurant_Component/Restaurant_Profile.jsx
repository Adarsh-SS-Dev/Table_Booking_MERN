import React, { useEffect, useState } from "react";
import Restaurant_Nav from "./Restaurant_Nav";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Restaurant_Profile = () => {
  const [menu, setMenu] = useState({
    starters: [],
    mainCourse: [],
    desserts: [],
  });

  const [restaurant, setRestaurant] = useState({
    name: "",
    ownername: "",
    address: "",
    phoneNo: "",
  });

  const hotelId = sessionStorage.getItem("restaurantid");
  const navigate = useNavigate();

  useEffect(() => {
    if (!hotelId) {
      navigate("/");
    } else {
      const name = sessionStorage.getItem("restaurantname");
      const ownername = sessionStorage.getItem("restaurantownername");
      const address = sessionStorage.getItem("restaurantadd");
      const phoneNo = sessionStorage.getItem("restaurantphNo");

      setRestaurant({ name, ownername, address, phoneNo });
      fetchMenuItems();
    }
  }, [hotelId, navigate]);

  const fetchMenuItems = async () => {
    try {
      const [starterRes, mainRes, dessertRes] = await Promise.all([
        axios.post("http://localhost:3001/viewstarters", { hotelid: hotelId }),
        axios.post("http://localhost:3001/viewmains", { hotelid: hotelId }),
        axios.post("http://localhost:3001/viewdesserts", { hotelid: hotelId }),
      ]);

      const starters = starterRes.data.map((item) => ({
        id: item._id,
        name: item.item,
        price: item.price,
      }));

      const mainCourse = mainRes.data.map((item) => ({
        id: item._id,
        name: item.mitem,
        price: item.mprice,
      }));

      const desserts = dessertRes.data.map((item) => ({
        id: item._id,
        name: item.ditem,
        price: item.dprice,
      }));

      setMenu({ starters, mainCourse, desserts });
    } catch (error) {
      console.error("Error fetching menu items:", error);
      toast.error("Failed to load menu items.");
    }
  };

  const handleUpdate = async (category, itemId) => {
    const item = menu[category].find((i) => i.id === itemId);
    try {
      const response = await axios.post("http://localhost:3001/updateMenuItem", {
        itemId,
        category,
        name: item.name,
        price: item.price,
      });

      if (response.status === 200) {
        toast.success(`Updated: ${item.name}`);
      }
    } catch (error) {
      toast.error("Failed to update item.");
    }
  };

  const handleDelete = async (category, itemId) => {
    const itemToDelete = menu[category].find((item) => item.id === itemId);
    const confirmDelete = window.confirm(`Delete ${itemToDelete.name}?`);

    if (!confirmDelete) return;

    try {
      const response = await axios.post("http://localhost:3001/deleteMenuItem", {
        itemId,
        category,
      });

      if (response.status === 200) {
        // Remove the deleted item from the menu state
        const updatedMenu = { ...menu };
        updatedMenu[category] = updatedMenu[category].filter(
          (item) => item.id !== itemId
        );
        setMenu(updatedMenu);

        toast.success(response.data.message); // Toast success message
      }
    } catch (error) {
      toast.error("Failed to delete item.");
    }
  };

  const renderTable = (category, items) => (
    <div>
      <h2 style={{ textTransform: "capitalize", marginTop: "20px" }}>
        {category}
      </h2>
      <table
        border={2}
        cellPadding={10}
        style={{ width: "100%", marginBottom: "30px" }}
      >
        <thead style={{ backgroundColor: "#f0f0f0" }}>
          <tr>
            <th>SI</th>
            <th>ITEM</th>
            <th>PRICE</th>
            <th>UPDATE</th>
            <th>DELETE</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, idx) => (
            <tr key={item.id}>
              <td>{idx + 1}</td>
              <td>
                <input
                  type="text"
                  value={item.name}
                  onChange={(e) => {
                    const updatedMenu = { ...menu };
                    updatedMenu[category][idx].name = e.target.value;
                    setMenu(updatedMenu);
                  }}
                />
              </td>
              <td>
                <input
                  type="number"
                  value={item.price}
                  onChange={(e) => {
                    const updatedMenu = { ...menu };
                    updatedMenu[category][idx].price = e.target.value;
                    setMenu(updatedMenu);
                  }}
                />
              </td>
              <td>
                <button
                  onClick={() => handleUpdate(category, item.id)}
                  style={{
                    padding: "5px 10px",
                    backgroundColor: "#4CAF50",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                  }}
                >
                  Update
                </button>
              </td>
              <td>
                <button
                  onClick={() => handleDelete(category, item.id)}
                  style={{
                    padding: "5px 10px",
                    backgroundColor: "#f44336",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                  }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const sidebarStyle = {
    width: "250px",
    display: "flex",
    flexDirection: "column",
    height: "100vh",
  };

  const containerStyle = {
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
    width: "800px",
    backgroundColor: "white",
    padding: "20px",
    borderRadius: "8px",
    overflowY: "auto",
    boxShadow: "0 4px 10px rgba(0,0,0,0.1)",
    textAlign: "center", // Center the profile text
  };

  return (
    <div style={{ display: "flex" }}>
      <div style={sidebarStyle}>
        <Restaurant_Nav />
      </div>

      <div style={containerStyle}>
        <div style={cardStyle}>
          <div className="card mb-3">
            <div className="card-header">Restaurant Profile</div>
            <div
              className="card-body"
              style={{ display: "flex", justifyContent: "center", alignItems: "center" }}
            >
              <div>
                <h4>{restaurant.name}</h4>
                <p>
                  <strong>Owner :</strong> {restaurant.ownername}
                </p>
                <p>
                  <strong>Address :</strong> {restaurant.address}
                </p>
                <p>
                  <strong>Phone No. :</strong> {restaurant.phoneNo}
                </p>
              </div>
            </div>
          </div>

          {renderTable("starters", menu.starters)}
          {renderTable("mainCourse", menu.mainCourse)}
          {renderTable("desserts", menu.desserts)}
        </div>
      </div>

      <ToastContainer position="top-right" autoClose={2000} hideProgressBar />
    </div>
  );
};

export default Restaurant_Profile;
