
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import './App.css';
import Home from './components/mainhome/Home';
import About from './components/mainhome/About';
import Nav from './components/mainhome/Nav';
import Review from './components/mainhome/Review';
import Login from './components/Customer_Component/Login';
import Registration from './components/Customer_Component/Registration';
import Reslogin from './components/Restaurant_Component/Reslogin';
import Resregistration from './components/Restaurant_Component/Resregistration';
import Adlogin from './components/Admin_Component/Adlogin';
import Customer_Nav from './components/Customer_Component/Customer_Nav';
import Customer_Review from './components/Customer_Component/Customer_Review';
import Weather from './components/Customer_Component/Weather';
import Customer_hotellist from './components/Customer_Component/Customer_hotellist';
import Restauranthome from './components/Restaurant_Component/Restauranthome';
import Profile from './components/Customer_Component/Profile';
import Restaurant_Profile from './components/Restaurant_Component/Restaurant_Profile';
import RestaurantDetails from './components/Customer_Component/RestaurantDetails';
import Restads from './components/Restaurant_Component/Restads';
import Admincus from './components/Admin_Component/Admincus';
import Adminnav from './components/Admin_Component/Adminnav';
import Adminres from './components/Admin_Component/Adminres';
import Review_Ads from './components/Admin_Component/Review_Ads';
import Menu from './components/Restaurant_Component/Menu';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
      <Routes>
        <Route path='/' element={<Home/>}/>
        <Route path='/about' element={<About/>}/>
        <Route path='/review' element={<Review/>}/>
        <Route path='/customerlogin' element={<Login/>}/>
        <Route path='/customerregistration' element={<Registration/>}/>
        <Route path='/restaurantlogin' element={<Reslogin/>}/>
        <Route path='/restaurantregistration' element={<Resregistration/>}/>
        <Route path='/adminlogin' element={<Adlogin/>}/>
        <Route path='/cusnav' element={<Customer_Nav/>}/>
        <Route path='/customerreview' element={<Customer_Review/>}/>
        <Route path='/weather' element={<Weather/>}/>
        <Route path='/hotellist' element={<Customer_hotellist/>}/>
        <Route path='/resthome' element={<Restauranthome/>}/>
        <Route path='/profile' element={<Profile/>}/>
        <Route path='/resprofile' element={<Restaurant_Profile/>}/>
        <Route path="/restaurant" element={<RestaurantDetails />} />
        <Route path='/resads' element={<Restads/>}/>
        <Route path='/admincus' element={<Admincus/>}/>
        <Route path='/adminres' element={<Adminres/>}/>
        <Route path='/review_ads' element={<Review_Ads/>}/>
        <Route path='/menu' element={<Menu/>}/>
        
      </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
