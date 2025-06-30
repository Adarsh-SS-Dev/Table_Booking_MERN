import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Nav from '../mainhome/Nav';

const Adlogin = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const validUsername = "Admin123";
    const validPassword = "123456";

    const handleLogin = (e) => {
        e.preventDefault();

        if (username === validUsername && password === validPassword) {
            // Simulated token (not real JWT)
            const fakeToken = 'admintoken123456';
            sessionStorage.setItem('adtoken', fakeToken);
            navigate('/admincus');
        }
        else {
            setError('Invalid username or password');
        }
    };

    return (
        <div>
            <Nav />
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", backgroundColor: "white" }}>
                <div style={{
                    width: "500px",
                    height: "500px",
                    backgroundColor: "white",
                    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
                    borderRadius: "10px",
                    display: "flex",
                    flexDirection: "column",
                }}>
                    <h1><u>ADMIN LOGIN</u></h1>

                    {error && <div style={{ color: 'red', padding: '10px' }}>{error}</div>}

                    <div style={{ display: "flex", flexDirection: "row", justifyContent: "space-between", gap: "20px", padding: "20px" }}>
                        <label>Username</label>
                        <input
                            type="text"
                            className="form-control"
                            id="username"
                            placeholder="Enter your username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                        />
                    </div>

                    <div style={{ display: "flex", flexDirection: "row", justifyContent: "space-between", gap: "20px", padding: "20px" }}>
                        <label>Password</label>
                        <input
                            type="password"
                            className="form-control"
                            id="password"
                            placeholder="Enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                    </div>

                    <div style={{ display: "flex", flexDirection: "row", justifyContent: "center", gap: "20px", padding: "20px" }}>
                        <button className="btn btn-success" type="submit" onClick={handleLogin}>Login</button>
                    </div>

                    <br />
                    <hr style={{ width: "80%", height: "2px", backgroundColor: "black", border: "none", margin: "10px auto" }} />
                </div>
            </div>
        </div>
    );
};

export default Adlogin;
