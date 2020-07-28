import twint

def ambilData(kataKunci, periode_dari,periode_sampai):
    config = twint.Config()
    
    config.Search = kataKunci
    config.Since = periode_dari
    config.Until = periode_sampai
    config.Lang = "id"
    config.Store_csv = True
    config.Output = "data-"+periode_dari+"-"+periode_sampai+".csv"
   
    return twint.run.Search(config)
