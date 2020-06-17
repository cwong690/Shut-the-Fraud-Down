<!-- <img alt="" src='' height="600px" width="1000px" align='center'> -->

# Shut the FRAUD Down!

### A fraud detection model with an interactive Flask app to stream events and automatically detect fraudulent cases!

![badge](https://img.shields.io/badge/last%20modified-may%20%202020-success)
![badge](https://img.shields.io/badge/status-in%20progress-yellow)

<a href="https://github.com/b-weintraub">Ben Weintraub</a> | <a href="https://github.com/cwong90">Cindy Wong</a> | <a href="https://github.com/tylerjwoods">Tyler Woods</a>

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
    - [EDA](#eda)
- [Models](#models)
- [Summary](#summary)
- [Notes](#notes)
- [Future Work](#future-work)

## Overview

<b>Premise:</b>
A new e-commerce site needs a data scientist to detect fraudsters. A machine learning model needs to be created. However, failures are not equal. False positives will decrease trust with consumers and false negatives will cost money.

The model does not predict a ground truth, but rather flags ones that have high potential.
The interactive portion of the web app allows users to see which cases are the top priorities to check and attributes about that case.


## Data Preparation

<details>
  <summary>
    <b> Dataset </b>  
  </summary>
</details>


### EDA

Fraudelent categories

   Channels vs Fraud       |  Delivery Method vs Fraud |     Gross Profits vs Fraud
:-------------------------:|:-------------------------:|:-------------------------:
![](images/channels_eda.png) |   ![](images/delivery_method_eda.png)|    ![gross profits](images/gross_profits_dummie.png)

   FB Published vs Fraud   |  Ticket Length vs Fraud   |     User Type vs Fraud
:-------------------------:|:-------------------------:|:-------------------------:
![](images/fb_published.png)|   ![](images/ticket_type_length.png)|    ![gross profits](images/user_type.png)

   Sale Duration vs Fraud  |  Gmail vs Fraud           |     Previous Payout vs Fraud
:-------------------------:|:-------------------------:|:-------------------------:
![](images/sale_duration2.png)|   ![](images/gmail_account_eda.png)|    ![gross profits](images/previous_payouts_eda.png)


<!-- <img alt="" src='' style='width: 600px;'> -->


## Models


<details>
    <summary>Logistic Regression</summary>
<!--     <img alt="" src=''> -->
</details>   

<details>
    <summary>Random Forest Classifier</summary>
<!--     <img alt="" src=''> -->
</details>

<details>
    <summary>XGBoost</summary>
<!--     <img alt="" src=''> -->
</details>

<details>
    <summary>Gradient Boosting</summary>
<!--     <img alt="" src=''> -->
</details>

## Summary

## Notes


## Future Work

- [ ] KNN
- [ ] Better Model
- [ ] Clean up files