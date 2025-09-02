using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace FaReNEW
{
    public partial class WebForm4 : System.Web.UI.Page
    {
        protected void Page_Load(object sender, EventArgs e)
        {
            Label2.Text = Session["UN"].ToString();
            Label4.Text = Session["PS"].ToString();
            Label6.Text = Session["EM"].ToString();
            Label8.Text = Session["MOB"].ToString();
            Label10.Text = Session["DOB"].ToString();

        }

        protected void LinkButton3_Click(object sender, EventArgs e)
        {
            Response.Redirect("Discover.html");
        }

        protected void LinkButton2_Click(object sender, EventArgs e)
        {
            Response.Redirect("About_us.html");
        }

        protected void LinkButton1_Click(object sender, EventArgs e)
        {
            string email = "facecappers@gmail.com";
            ClientScript.RegisterStartupScript(this.GetType(), "mailto", "parent.location='mailto:" + email + "'", true);
        }

        protected void Button1_Click(object sender, EventArgs e)
        {
            Response.Redirect("Home.aspx");
        }
    }
}