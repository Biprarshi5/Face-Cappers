using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using System.Data;
using System.Data.SqlClient;
using System.Configuration;

namespace FaReNEW
{
    public partial class WebForm6 : System.Web.UI.Page
    {
        SqlConnection conn = new SqlConnection();
        protected void Page_Load(object sender, EventArgs e)
        {
            conn.ConnectionString = (@"Data Source=LAPTOP-4LQIPA5M;Initial Catalog=face_recog;user id=sa;password=avi@123");
            conn.Open();
            
        }
       

        protected void GridView1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        

        protected void GridView2_SelectedIndexChanged(object sender, EventArgs e)
        {
            
        }
        protected void GridView1_RowDeleting(object sender, GridViewDeleteEventArgs e)
        { 
            GridViewRow row = (GridViewRow)GridView1.Rows[e.RowIndex];
            SqlCommand cmd = new SqlCommand("delete from [Registration] where [Mob]=@Mob",conn) ;
            GridView1.DataBind();
            cmd.ExecuteNonQuery();
            
        }
        protected void LinkButton2_Click(object sender, EventArgs e)
        {
            Response.Redirect("About_us.html");
        }
        protected void Button1_Click1(object sender, EventArgs e)
        {

            Response.Redirect("Home.aspx");
        }

        protected void LinkButton3_Click(object sender, EventArgs e)
        {
            Response.Redirect("Discover.html");
        }

        protected void LinkButton1_Click(object sender, EventArgs e)
        {
            string email = "facecappers@gmail.com";
            ClientScript.RegisterStartupScript(this.GetType(), "mailto", "parent.location='mailto:" + email + "'", true);
        }
    }
}