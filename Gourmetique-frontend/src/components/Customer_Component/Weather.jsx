import React, { useEffect, useState } from "react";
import axios from "axios";
import Customer_Nav from "./Customer_Nav";
import { useNavigate } from "react-router-dom";

const Weather = () => {
  const [city, setCity] = useState("");  // State for city input
  const [weather, setWeather] = useState(null); // State for weather data
  const [error, setError] = useState(""); // State for error messages
  const navigate = useNavigate(); // For navigation
  const customerId = sessionStorage.getItem("customerid");

  useEffect(() => {
    if (!customerId) {
      navigate('/'); // Navigate to login if restaurantid is not in sessionStorage
    }
  }, [customerId, navigate]);

  const fetchWeather = async () => {
    if (!city) {
      setError("Please enter a city name.");
      return;
    }
    setError("");

    try {
      const response = await axios.get(
        `https://geocoding-api.open-meteo.com/v1/search?name=${city}&count=1`
      );

      if (response.data.results) {
        const { latitude, longitude } = response.data.results[0];
        const weatherResponse = await axios.get(
          `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`
        );
        setWeather(weatherResponse.data.current_weather);
      } else {
        setError("City not found. Please try again.");
        setWeather(null);
      }
    } catch (err) {
      setError("Error fetching data.");
      setWeather(null);
    }
  };

  return (
    <div>
      <Customer_Nav />

      <div style={styles.container}>
        <h1>Weather App</h1>
        <input
          type="text"
          placeholder="Enter city name"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          style={styles.input}
        />
        <button onClick={fetchWeather} style={styles.button}>
          Search
        </button>
        {error && <p style={styles.error}>{error}</p>}
        {weather && (
          <div style={styles.weatherInfo}>
            <h2>Temperature: {weather.temperature}°C</h2>
            <p>Wind Speed: {weather.windspeed} m/s</p>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: { textAlign: "center", padding: "20px", fontFamily: "Arial" },
  input: { padding: "10px", fontSize: "16px", marginRight: "10px" },
  button: { padding: "10px 15px", fontSize: "16px", cursor: "pointer" },
  error: { color: "red", marginTop: "10px" },
  weatherInfo: { marginTop: "20px", padding: "10px", border: "1px solid #ddd" },
};

export default Weather;
