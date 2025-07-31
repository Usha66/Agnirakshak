import React from "react";
const Footer = () => {
  const currentYear = new Date().getFullYear();

  return <div className="footer">Fire Fortifiers 22.1- &copy;{currentYear}</div>;
};

export default Footer;
