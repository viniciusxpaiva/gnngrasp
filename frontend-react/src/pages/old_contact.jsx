import React from 'react';
import BaseLayout from '../components/layout/base';

const Contact = () => {
	return (
		<BaseLayout>
			<div className="container-fluid bg-light-dark text-white mt-0 py-4"  id="help-submit">
				<div className="row justify-content-center" >
					<div class="col-md-12 text-center">
						<h6 className="display-6 text-light">Contact</h6>
					</div>
					</div>
				</div>


					<div class="container mt-4">
						<div class="row">
						    <h3>Contact details</h3>
						</div>
						<hr/>
						<div class="row ">
						<div class="col-md-5" style={{fontSize:"18px"}}>
							<div class="crad border-light">
							<div class="card-body">
								<h5 class="card-title"><a className="text-decoration-none" href="http://lattes.cnpq.br/0899817111748167" title="Lattes curriculum" target="_blank" rel="noopener noreferrer"><b>Sabrina Silveira</b></a></h5>
                                <br></br>
                                <p>sabrina@ufv.br<br />ORCID: <a className="text-decoration-none" href="https://orcid.org/0000-0002-4723-2349" target="_blank" rel="noopener noreferrer">0000-0002-4723-2349</a></p>
                                <p></p>
								<br />
								<h5 class="card-title"><a className="text-decoration-none" href="http://lattes.cnpq.br/2889736880174687" title="Lattes curriculum" target="_blank" rel="noopener noreferrer"><b>Vinícius Paiva</b></a></h5>
                                <br></br>
                                <p>vinicius.d.paiva@ufv.br<br />ORCID: <a className="text-decoration-none" href="https://orcid.org/0000-0002-4411-1875" target="_blank" rel="noopener noreferrer">0000-0002-4411-1875</a></p>
                                <p></p>
								<br />
							</div>
							</div>
						</div>
						<div class="col-md-7" style={{fontSize:"18px"}}>
							<div class="crad border-light ">
							<div class="card-body">
								<h5 class="card-title"><a className="text-decoration-none" href="https://homepages.dcc.ufmg.br/~sabrinas" target="_blank" rel="noopener noreferrer"><b>LaBio Laboratory of Bioinformatics, Visualization and Systems</b></a></h5>
                                <br></br>
                                <p>Department of Computer Science
                                    <br />Universidade Federal de Viçosa
                                    <br />Viçosa - Minas Gerais - Brazil
                                    <br />36570-900
                                    <br />+55 (31) 3612-6359
                                </p>
							</div>
							</div>
						</div>
                        <hr />
						</div>
						
					</div>
					

		</BaseLayout>
	);
};
export default Contact;
