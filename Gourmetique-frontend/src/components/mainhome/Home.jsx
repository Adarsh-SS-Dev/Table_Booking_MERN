import { Carousel } from 'bootstrap/dist/js/bootstrap.min'
import React from 'react'
import Nav from './Nav'
import Carousel1 from './Carousel1'
import Details from './Details'

const Home = () => {
  return (
    // <div className="container" style={{backgroundColor:"yellow"}}>
        <div className="row">
        <div className="col col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12 col-xxl-12" style={{marginBottom:"20px"}}>
            <Nav/>
        </div>
        <div className="col col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12 col-xxl-12" style={{marginBottom:"20px"}}>
            <Carousel1 interval={1000} pause="hover"/>
        </div>
        <div className="col col-12 col-sm-12 col-md-12 col-lg-12 col-xl-12 col-xxl-12" style={{marginBottom:"20px"}}>
            {/* <Details/> */}
        </div>
        </div>
       
    // </div>
   
  )
}

export default Home