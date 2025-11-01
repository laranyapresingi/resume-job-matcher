def get_roles():
    return ["SDE", "Data Scientist"]

def get_job_files():
    return {
        "SDE": {
            "Backend Developer": "jds_clean/sde_roles/backend_devp.txt",
            "Frontend Developer": "jds_clean/sde_roles/frontend_devp.txt",
            "Software Engineer" : "jds_clean/sde_roles/sde.txt"
        },
        "Data Scientist": {
            "Data Scientist": "jds_clean/ds_roles/data_scientist.txt",
            "ML Engineer": "jds_clean/ds_roles/ml_engineer.txt",
            "Business Analyst" : "jds_clean/ds_roles/business_analyst.txt",
            "Data Analyst" : "jds_clean/ds_roles/data_analyst.txt"
        }
    }

def load_jd_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()