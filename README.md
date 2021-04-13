# price-prediction
## how to use
 - git clone https://github.com/matsui222/price_prediction_ver2_tmp.git
 - docker build -t price_pred_tmp:1.0 .
 - docker run --rm -it --publish 5000:5000 --detach --env PORT=5000 price_pred_tmp:1.0 bash
 - docker run --rm price_pred_tmp:1.0 python test_tiger_ver2.py '{ "registered_monthly_date": "2020-01-30 00:00:00", "plan_id": 1, "payment_type": 2, "date_of_birth": "1996-01-30 00:00:00", "user_fi_count": 3, "purchase_cnt": 5, "item_data": { "39942": { "seasons": { "summer": false, "winter": true, "late_autumn": true, "late_spring": false, "early_autumn": true, "early_spring": true }, "ac_color": 5, "category_id": 5, "retail_price": 5000, "item_fi_count": 43, "max_sale_price": 1000, "max_sale_rate": 0.22 }, "52321": { "seasons": { "summer": false, "winter": true, "late_autumn": true, "late_spring": false, "early_autumn": true, "early_spring": true }, "ac_color": 2, "category_id": 3, "retail_price": 8000, "item_fi_count": 54, "max_sale_price": 2000, "max_sale_rate": 0.2 } }, "coupon_discount_price": [ 1000, 800, 100 ], "coupon_discount_rate": [ 0.1, 0.05, 0.3, 0.001 ] }'
