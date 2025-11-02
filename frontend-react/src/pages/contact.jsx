import React from 'react';
import BaseLayout from '../components/layout/base';
import Stack from "@mui/material/Stack";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import PropTypes from "prop-types";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";

function a11yProps(index) {
	return {
		id: `simple-tab-${index}`,
		"aria-controls": `simple-tabpanel-${index}`,
	};
}

CustomTabPanel.propTypes = {
	children: PropTypes.node,
	index: PropTypes.number.isRequired,
	value: PropTypes.number.isRequired,
};

function CustomTabPanel(props) {
	const { children, value, index, ...other } = props;

	return (
		<div
			role="tabpanel"
			hidden={value !== index}
			id={`simple-tabpanel-${index}`}
			aria-labelledby={`simple-tab-${index}`}
			{...other}
		>
			{value === index && (
				<Box sx={{ p: 3 }}>
					<Typography>{children}</Typography>
				</Box>
			)}
		</div>
	);
}

const Contact = () => {
	return (
		<BaseLayout>
			<div className="container-fluid bg-light-dark text-white mt-0 py-4" id="help-submit">
				<div className="row justify-content-center" >
					<div class="col-md-12 text-center">
						<h6 className="display-6 text-light">Contact</h6>
					</div>
				</div>
			</div>
			<div className="container">

				<Box
					sx={{
						borderBottom: 1,
						borderColor: "divider",
						display: "flex",
						justifyContent: "center",
					}}
				>

					<Tabs
						aria-label="basic tabs example"
						variant="scrollable"
						scrollButtons="auto"
						value={0}
					>
						<Tab label={"Contact details"} {...a11yProps(0)} />
					</Tabs>
				</Box>
				<CustomTabPanel value={0} index={0}>
					<Card variant="outlined">
						<Box sx={{ p: 2 }}>
							<div className="row mt-2 mb-2">
								<div className="col-12">
									<Stack
										direction="row"
										justifyContent="space-between"
										alignItems="center"
									>
										<Typography gutterBottom variant="h5" component="div">
											<a className="text-decoration-none" href="https://homepages.dcc.ufmg.br/~sabrinas" target="_blank" rel="noopener noreferrer">LaBio Laboratory of Bioinformatics, Visualization and Systems</a>
										</Typography>
									</Stack>
									<Typography color="text.secondary" variant="body1">
										<p>Department of Computer Science
											<br />Universidade Federal de Viçosa
											<br />Viçosa - Minas Gerais - Brazil
											<br />36570-900
											<br />+55 (31) 3612-6359
										</p>
									</Typography>
									<Stack
										direction="row"
										justifyContent="space-between"
										alignItems="center"
										marginTop={5}

									>
										<Typography gutterBottom variant="h5" component="div">
											<a className="text-decoration-none" href="http://lattes.cnpq.br/0899817111748167" title="Lattes curriculum" target="_blank" rel="noopener noreferrer">Sabrina Silveira</a>
										</Typography>
									</Stack>
									<Typography color="text.secondary" variant="body1">
										sabrina@ufv.br<br />
										ORCID: <a className="text-decoration-none" href="https://orcid.org/0000-0002-4723-2349" target="_blank" rel="noopener noreferrer">0000-0002-4723-2349</a><br />
									</Typography>
									<Stack
										direction="row"
										justifyContent="space-between"
										alignItems="center"
										marginTop={5}
									>
										<Typography gutterBottom variant="h5" component="div">
											<a className="text-decoration-none" href="http://lattes.cnpq.br/2889736880174687" title="Lattes curriculum" target="_blank" rel="noopener noreferrer">Vinícius Paiva</a>
										</Typography>
									</Stack>
									<Typography color="text.secondary" variant="body1">
										vinicius.d.paiva@ufv.br<br />
										ORCID: <a className="text-decoration-none" href="https://orcid.org/0000-0002-4411-1875" target="_blank" rel="noopener noreferrer">0000-0002-4411-1875</a><br />
									</Typography>
								</div>
							</div>
						</Box>
						<Divider />
					</Card>
				</CustomTabPanel>
			</div>
		</BaseLayout>
	);
};
export default Contact;
