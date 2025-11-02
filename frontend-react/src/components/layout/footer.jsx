import React from "react";
import { MDBFooter, MDBContainer } from "mdb-react-ui-kit";

const Footer = () => {
  return (
    <>
      <footer
        style={{
          position: "absolute",
          bottom: 0,
          width: "100%",
          height: "0.5rem",
        }}
      >
        <MDBFooter
          className="text-center text-white"
          style={{ backgroundColor: "#f1f1f1" }}
        >
          <MDBContainer className="pt-4">
            <section className="mb-3 text-dark">
              <div className="row">
                <span>© 2024 Copyright BENDER DB</span>
                <a
                  className="text-decoration-none mt-2"
                  href="https://homepages.dcc.ufmg.br/~sabrinas"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  LaBio Laboratory of Bioinformatics, Visualization and Systems
                </a>
              </div>
            </section>
          </MDBContainer>

          <div
            className="text-center text-dark p-3"
            style={{ backgroundColor: "rgba(0, 0, 0, 0.2)" }}
          >
            <div className="container">
              <div className="row">
                <div className="col">
                  {" "}
                  <a
                    href="https://www.ufv.br/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <img
                      src="/img/ufv.png"
                      className="img-fluid"
                      style={{
                        maxWidth: "100%",
                        maxHeight: "60px",
                        width: "auto",
                        height: "auto",
                      }}
                      alt="ufv-logo"
                    />
                  </a>
                </div>
                <div className="col">
                  <a
                    href="https://www.unimelb.edu.au/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <img
                      src="/img/uni_melb.png"
                      className="img-fluid"
                      style={{
                        maxWidth: "100%",
                        maxHeight: "60px",
                        width: "auto",
                        height: "auto",
                      }}
                      alt="uni-melb-logo"
                    />
                  </a>
                </div>
                <div className="col">
                  <a
                    href="https://unifei.edu.br/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <img
                      src="/img/unifei2.png"
                      className="img-fluid"
                      style={{
                        maxWidth: "100%",
                        maxHeight: "60px",
                        width: "auto",
                        height: "auto",
                      }}
                      alt="unifei-logo"
                    />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </MDBFooter>
      </footer>
      <script
        type="text/javascript"
        src="//www.privacypolicies.com/public/cookie-consent/4.0.0/cookie-consent.js"
        charset="UTF-8"
      ></script>
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js"></script>
    </>
  );
};
export default Footer;
