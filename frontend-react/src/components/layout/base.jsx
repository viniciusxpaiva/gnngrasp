import Footer from "./footer";
import ResponsiveAppBar from "./new_navbar";

const BaseLayout = ({children}) => {
    return (
        <>
            <title>GNN-GRaSP</title>
            <div style={{position: "relative", minHeight:"100vh"}}>
                <div style={{paddingBottom: "5.5rem"}}>
                    <ResponsiveAppBar></ResponsiveAppBar>
                {children}
                </div>
            <Footer></Footer>
            </div>
        </>

    )

}
export default BaseLayout;