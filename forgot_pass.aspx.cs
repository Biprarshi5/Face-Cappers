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
    public partial class forgot_pass : System.Web.UI.Page
    {
        SqlConnection conn = new SqlConnection(@"Data Source=LAPTOP-4LQIPA5M;Initial Catalog=face_recog;user id=sa;password=avi@123");
        SqlCommand cmd = new SqlCommand();
        protected void Page_Load(object sender, EventArgs e)
        {
            
        }

        protected void TextBox5_TextChanged1(object sender, EventArgs e)
        {

        }

        protected void TextBox6_TextChanged(object sender, EventArgs e)
        {

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

        protected void Button1_Click1(object sender, EventArgs e)
        {
            cmd.Connection = conn;
            cmd.CommandText = "select *from Registration where User_Name='" + TextBox1.Text + "' and Email='" + TextBox4.Text + "' and Mob='" + TextBox5.Text + "' and DOB='" + TextBox6.Text + "'";

            conn.Open();
            SqlDataReader dr = cmd.ExecuteReader();
            if (dr.Read())
            {
                if (dr.HasRows == true)
                {
                    
                   

                    Session["user"] = dr[0].ToString();
                    Session["EM"] = dr[1].ToString();
                    Session["Mob"] = dr[2].ToString();
                    Session["DOB"] = dr[3].ToString();
                    TextBox2.Visible = true;
                    TextBox3.Visible = true;
                    
                }
            }
           

            conn.Close();
        }

        protected void Button2_Click(object sender, EventArgs e)
        {
            conn.Close();
            conn.Open();
            SqlCommand cd = new SqlCommand("update [Registration] set [Password] ='"+TextBox2.Text+"'"+ " where [Mob] = '"+ TextBox5.Text+"'", conn);
            cd.ExecuteNonQuery();
            conn.Close();

            Session["UN"] = TextBox1.Text.ToString();
            Session["PS"] = TextBox2.Text.ToString();
            Session["EM"] = TextBox4.Text.ToString();
            Session["MOB"] = TextBox5.Text.ToString();
            Session["DOB"] = TextBox6.Text.ToString();
            Response.Redirect("Regissucc.aspx");
        }
    }

       
    }
