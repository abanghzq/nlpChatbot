import mysql.connector

# Connect to your MySQL database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='chatbot_project'
)
cursor = conn.cursor()

# Your data
intents = [
    {
      "tag": "account",
      "patterns": [
        "How can I create an account?",
        "Can you guide me on signing up?",
        "What are the steps to register an account?",
        "Tell me about the account creation process.",
        "Where do I go to create an account?",
        "Is there a specific button for account creation?",
        "What's the procedure for signing up?",
        "Can you walk me through the account registration steps?",
        "How do I go about making an account?",
        "Is there a tutorial for creating an account?"
      ],
      "responses": [
        "To create an account, click on the 'Sign Up' button on the top right corner of our website and follow the instructions to complete the registration process."
      ]
    },
    {
      "tag": "payments",
      "patterns": [
        "What payment methods do you accept?",
        "Can I use PayPal for payment?",
        "What credit cards are accepted?",
        "Are debit cards a valid payment option?",
        "Do you support other payment methods?",
        "Can I pay with my bank account?",
        "Are there restrictions on payment methods?",
        "Which forms of payment are valid?",
        "Do you accept digital wallets?",
        "What are the approved payment options?"
      ],
      "responses": [
        "We accept major credit cards, debit cards, and PayPal as payment methods for online orders."
      ]
    },
    {
      "tag": "tracking",
      "patterns": [
        "How can I track my order?",
        "Where can I find information about my order?",
        "What's the process for tracking my order?",
        "Can I track my shipment?",
        "Where do I go to check my order status?",
        "Is there a specific order tracking page?",
        "What details do I need to track my order?",
        "Can you guide me on tracking my package?",
        "Are there notifications for order tracking?",
        "How do I stay updated on my order's status?"
      ],
      "responses": [
        "You can track your order by logging into your account and navigating to the 'Order History' section. There, you will find the tracking information for your shipment."
      ]
    },
    {
      "tag": "refund",
      "patterns": [
        "What is your return policy?",
        "Can you explain the conditions for returns?",
        "Tell me about your return process.",
        "How long do I have to return a product?",
        "What are the terms of your return policy?",
        "Is there a specific window for returns?",
        "What items are eligible for returns?",
        "Can I return a product without the original packaging?",
        "What happens if I want to return an item?",
        "How do I initiate a return?"
      ],
      "responses": [
        "Our return policy allows you to return products within 30 days of purchase for a full refund, provided they are in their original condition and packaging. Please refer to our Returns page for detailed instructions."
      ]
    },
    {
      "tag": "cancel",
      "patterns": [
        "Can I cancel my order?",
        "Is order cancellation possible?",
        "What's the process for canceling an order?",
        "How do I go about canceling my order?",
        "Is there a time limit for order cancellation?",
        "Can I change my mind after placing an order?",
        "What steps should I take to cancel my order?",
        "Is there a cancellation fee?",
        "Can I cancel a part of my order?",
        "Are there restrictions on order cancellation?"
      ],
      "responses": [
        "You can cancel your order if it has not been shipped yet. Please contact our customer support team with your order details, and we will assist you with the cancellation process."
      ]
    },
    {
      "tag": "shipping",
      "patterns": [
        "How long does shipping take?",
        "What is the usual shipping duration?",
        "Can you provide an estimate of the delivery time?",
        "When can I expect my order to arrive?",
        "What factors influence shipping times?",
        "Is there an expedited shipping option?",
        "Do you offer same-day shipping?",
        "Can I track my shipment in real-time?",
        "Are there delays in shipping?",
        "What's the average delivery time?"
      ],
      "responses": [
        "Shipping times vary depending on the destination and the shipping method chosen. Standard shipping usually takes 3-5 business days, while express shipping can take 1-2 business days."
      ]
    },
    {
      "tag": "internationalShipping",
      "patterns": [
        "Do you offer international shipping?",
        "Can I get your products delivered internationally?",
        "Is international shipping an option?",
        "Do you ship to other countries?",
        "Which countries do you ship to?",
        "What are the international shipping rates?",
        "How do I know if my country is eligible for shipping?",
        "Is there a separate process for international orders?",
        "What's the average delivery time for international orders?",
        "Are there additional charges for international shipping?"
      ],
      "responses": [
        "Yes, we offer international shipping to select countries. The availability and shipping costs will be calculated during the checkout process based on your location."
      ]
    },
    {
      "tag": "damaged/lost",
      "patterns": [
        "What should I do if my package is lost or damaged?",
        "How do I handle lost or damaged packages?",
        "What's the protocol for dealing with damaged shipments?",
        "Can you guide me on lost package resolution?",
        "Is there insurance for lost or damaged packages?",
        "What steps should I take if my package is missing?",
        "How can I report a damaged package?",
        "Do I need to provide evidence of package damage?",
        "Is there a timeframe for reporting lost packages?",
        "What is the typical resolution for lost or damaged items?"
      ],
      "responses": [
        "If your package is lost or damaged during transit, please contact our customer support team immediately. We will initiate an investigation and take the necessary steps to resolve the issue."
      ]
    },
    {
      "tag": "shippingAddress",
      "patterns": [
        "Can I change my shipping address after placing an order?",
        "Is it possible to update the shipping address?",
        "What's the procedure for changing the delivery address?",
        "How do I go about modifying the shipping details?",
        "Can I edit the shipping address on my order?",
        "Are there restrictions on address changes?",
        "Is there a timeframe for updating the shipping details?",
        "Can I change the delivery address for part of my order?",
        "What steps should I take to update my address?",
        "Is there a fee for changing the shipping address?"
      ],
      "responses": [
        "If you need to change your shipping address, please contact our customer support team as soon as possible. We will do our best to update the address if the order has not been shipped yet."
      ]
    },
    {
      "tag": "wrapping",
      "patterns": [
        "Do you offer gift wrapping services?",
        "Is gift wrapping available for orders?",
        "Can I request gift wrapping for my purchase?",
        "What's the process for adding gift wrapping?",
        "Is there an extra charge for gift wrapping?",
        "Can I customize the gift wrapping?",
        "Are there specific occasions for gift wrapping?",
        "Do you include a gift message with wrapping?",
        "Can I see examples of your gift wrapping?",
        "What types of items can be gift wrapped?"
      ],
      "responses": [
        "Yes, we offer gift wrapping services for an additional fee. During the checkout process, you can select the option to add gift wrapping to your order."
      ]
    },
    {
      "tag": "priceMatching",
      "patterns": [
        "What is your price matching policy?",
        "Do you match prices from other sellers?",
        "Can I get a price match on a product?",
        "What's the process for price matching?",
        "Are there specific criteria for price matching?",
        "Do you match both online and offline prices?",
        "Is there a timeframe for requesting a price match?",
        "What information do I need for a price match request?",
        "Are there any restrictions on the items eligible for price matching?",
        "How do I submit a price match request?"
      ],
      "responses": [ 
        "We have a price matching policy where we will match the price of an identical product found on a competitor's website. Please contact our customer support team with the details of the product and the competitor's offer."
      ]
    },
    {
      "tag": "orderByPhone",
      "patterns": [
        "Can I order by phone?",
        "Do you accept orders over the phone?",
        "Is phone ordering available?",
        "What's the process for placing an order by phone?",
        "Can I speak to a representative to place an order?",
        "Is there a dedicated hotline for phone orders?",
        "Are there specific hours for phone orders?",
        "What information do I need for a phone order?",
        "Can I modify my order placed by phone?",
        "Do you provide order confirmations for phone orders?"
      ],
      "responses": [
        "Unfortunately, we do not accept orders over the phone. Please place your order through our website for a smooth and secure transaction."
      ]
    },
    {
      "tag": "security",
      "patterns": [
        "Are my personal and payment details secure?",
        "What security measures protect my information?",
        "Can I trust the security of your website?",
        "How do you ensure the privacy of customer details?",
        "Are there encryption protocols in place?",
        "Is it safe to save payment details on my account?",
        "What steps do you take to prevent data breaches?",
        "Do you comply with industry standards for data security?",
        "Are there additional security features for accounts?",
        "Can I review the security measures in place?"
      ],
      "responses": [
        "Yes, we take the security of your personal and payment details seriously. We use industry-standard encryption and follow strict security protocols to ensure your information is protected."
      ]
    },
    {
      "tag": "priceAdjustments",
      "patterns": [
        "What is your price adjustment policy?",
        "Do you offer price adjustments after purchase?",
        "Can I get a refund if the price drops after I buy?",
        "Is there a timeframe for price adjustments?",
        "What's the process for requesting a price adjustment?",
        "Do you honor price drops within a certain period?",
        "Are all items eligible for price adjustments?",
        "Can I request multiple price adjustments?",
        "What information is needed for a price adjustment request?",
        "How will I be refunded for a price adjustment?"
      ],
      "responses": [
        "If a product you purchased goes on sale within 7 days of your purchase, we offer a one-time price adjustment. Please contact our customer support team with your order details to request the adjustment."
      ]
    },
    {
      "tag": "loyalty",
      "patterns": [
        "Do you have a loyalty program?",
        "Can I join your loyalty program?",
        "How does your loyalty program work?",
        "What are the benefits of your loyalty program?",
        "Is there a fee to join the loyalty program?",
        "How can I earn points in your loyalty program?",
        "Are there different membership tiers?",
        "Can I redeem points for any product?",
        "What's the process for joining the loyalty program?",
        "Do you offer exclusive rewards for loyalty members?"
      ],
      "responses": [
        "Yes, we have a loyalty program where you can earn points for every purchase. These points can be redeemed for discounts on future orders. Please visit our website to learn more and join the program."
      ]
    },
    {
      "tag": "orderGuest",
      "patterns": [
        "Can I order without creating an account?",
        "Is guest checkout available?",
        "Do I need to register to place an order?",
        "Can I make a purchase without signing up?",
        "Is there a guest option for ordering?",
        "Do I have to create an account for each order?",
        "What are the benefits of creating an account?",
        "Can I still track my order as a guest?",
        "Is there a quick checkout option?",
        "How do I proceed with checkout without registering?"
      ],
      "responses": [
        "Yes, you can place an order as a guest without creating an account. However, creating an account offers benefits such as order tracking and easier future purchases."
      ]
    },
    {
      "tag": "wholesale",
      "patterns": [
        "Do you offer bulk or wholesale discounts?",
        "Can I get a discount for bulk purchases?",
        "Is there a wholesale pricing option?",
        "Do you have special rates for large orders?",
        "Can I negotiate pricing for bulk quantities?",
        "What's the process for bulk ordering?",
        "Are there specific products eligible for wholesale discounts?",
        "Do you have a minimum order quantity for wholesale?",
        "Can I request a quote for a bulk order?",
        "Are there additional benefits for wholesale buyers?"
      ],
      "responses": [
        "Yes, we offer bulk or wholesale discounts for certain products. Please contact our customer support team or visit our Wholesale page for more information and to discuss your specific requirements."
      ]
    },
    {
      "tag": "change",
      "patterns": [
        "Can I change or cancel an item in my order?",
        "Is it possible to modify an order item?",
        "Can I swap items in my order?",
        "What's the process for changing order items?",
        "Is there a timeframe for modifying an order?",
        "Can I remove an item from my order?",
        "Do you allow item substitutions after order placement?",
        "Can I cancel part of my order?",
        "How do I edit my order contents?",
        "Are there restrictions on changing order items?"
      ],
      "responses": [
        "If you need to change or cancel an item in your order, please contact our customer support team as soon as possible. We will assist you with the necessary steps."
      ]
    },
    {
      "tag": "review",
      "patterns": [
        "How do I share my thoughts on a product?",
        "Is there a feedback option for purchased items?",
        "Tell me about the process for reviewing a product.",
        "Where can I write a review for a product?",
        "Do you have a rating system for products?",
        "Can I provide feedback on products?",
        "What's the procedure for leaving a review?",
        "Is there a dedicated review section on your website?",
        "How do I rate a product on your site?",
        "Are there guidelines for writing product reviews?"
      ],
      "responses": [
        "To leave a product review, navigate to the product page on our website and click on the 'Write a Review' button. You can share your feedback and rating based on your experience with the product."
      ]
    },
    {
      "tag": "moreThanOnePromoCode",
      "patterns": [
        "Is it possible to apply more than one promo code per order?",
        "Can I combine multiple promo codes during checkout?",
        "What's the limit for promo codes on a single order?",
        "Do you allow using multiple discount codes together?",
        "How many promo codes can I use in one order?",
        "Can I stack promo codes for additional discounts?",
        "Is there a restriction on the number of promo codes per order?",
        "What happens if I enter multiple promo codes?",
        "Can I enter more than one discount code at checkout?",
        "Are there any limitations on using promo codes concurrently?"
      ],
      "responses": [
        "Usually, only one promo code can be applied per order. During the checkout process, enter the promo code in the designated field to apply the discount to your order."
      ]
    },
    {
      "tag": "receiveWrongItem",
      "patterns": [
        "How do I handle receiving the incorrect item?",
        "What steps should I take for a wrong item in my order?",
        "Is there a procedure for receiving the wrong product?",
        "How to address receiving an item I didn't order?",
        "What's the process for dealing with incorrect orders?",
        "Can you guide me on receiving the wrong item?",
        "What if the item I receive doesn't match my order?",
        "Is there a customer support process for wrong items?",
        "How can I return an item I didn't order?",
        "What's the policy for correcting wrong items in an order?"
      ],
      "responses": [
        "If you receive the wrong item in your order, please contact our customer support team immediately. We will arrange for the correct item to be shipped to you and assist with returning the wrong item."
      ]
    },
    {
      "tag": "fasterShipping",
      "patterns": [
        "Is there an option for faster shipping?",
        "Can I choose expedited delivery for my order?",
        "Tell me about the expedited shipping choices.",
        "What are the options for quicker delivery?",
        "Do you provide faster shipping methods?",
        "Can I expedite the shipping process?",
        "Is there a premium service for faster shipping?",
        "How can I get my order delivered more quickly?",
        "What's the procedure for selecting expedited shipping?",
        "Are there any additional costs for expedited shipping?"
      ],
      "responses": [
        "Yes, we offer expedited shipping options for faster delivery. During the checkout process, you can select the desired expedited shipping method."
      ]
    },
    {
      "tag": "newsletter",
      "patterns": [
        "What content is included in your email newsletter?",
        "Can you provide more details about your newsletter content?",
        "Tell me about the information shared in your email updates.",
        "What updates can I expect in your email newsletter?",
        "Do your newsletters cover only product releases?",
        "Is the newsletter solely about new products?",
        "Can you elaborate on the topics covered in your newsletter?",
        "What kind of information is shared in your email updates?",
        "Are there specific themes in your newsletter content?",
        "Can you describe the content of your email updates?"
      ],
      "responses": [
        "Our email newsletter provides updates on new product releases, exclusive offers, and helpful tips related to our products. You can subscribe to our newsletter on our website."
      ]
    },
    {
      "tag": "returnChangeMind",
      "patterns": [
        "Is there a timeframe for returning a product if I change my mind?",
        "What's the time limit for returning a product due to a change of mind?",
        "Can I return a product for changing my mind after a specific period?",
        "Is there a deadline for changing my mind and returning a product?",
        "What is the allowed time frame for changing my mind and returning a product?",
        "Do you have a specific policy duration for changing my mind and returning a product?",
        "Can I still return a product if I changed my mind after a certain period?",
        "Is the return policy for changing my mind time-sensitive?",
        "What happens if I decide to return a product after a while due to a change of mind?",
        "Do you impose a time limit for returning a product if I change my mind?"
      ],
      "responses": [
        "Yes, you can return a product if you changed your mind. Please ensure the product is in its original condition and packaging, and refer to our return policy for instructions."
      ]
    },
    {
      "tag": "liveChat",
      "patterns": [
        "How do I access live chat support on your website?",
        "Is live chat support easily accessible on your site?",
        "What's the process for initiating a live chat on your website?",
        "Can you guide me on using live chat support during business hours?",
        "Where can I find the live chat icon for customer support?",
        "Is the live chat feature available throughout your business hours?",
        "Can you provide instructions on how to start a live chat with your team?",
        "Is there a specific location for accessing live chat on your site?",
        "What steps should I follow to engage in live chat support?",
        "Where exactly is the live chat icon located on your website?"
      ],
      "responses": [
        "Yes, we offer live chat support on our website during our business hours. Look for the chat icon in the bottom right corner to initiate a chat with our customer support team."
      ]
    },
    {
      "tag": "shippingGift",
      "patterns": [
        "Can I customize the order when purchasing a product as a gift?",
        "Is there an option to personalize the order for a gift purchase?",
        "How can I add a personal touch when ordering a product as a gift?",
        "Do you provide customization for products ordered as gifts?",
        "Is there a way to include a personalized message with a gift order?",
        "Can I request special packaging for a product ordered as a gift?",
        "What customization options are available for gift orders?",
        "Can I choose specific wrapping or packaging for a gift order?",
        "Do you offer any customization features for gift purchases?",
        "How flexible is the ordering process when buying a product as a gift?"
      ],
      "responses": [
        "Yes, you can order a product as a gift and have it shipped directly to the recipient. During the checkout process, you can enter the recipient's shipping address."
      ]
    },
    {
      "tag": "discountCode",
      "patterns": [
        "What troubleshooting steps can I take if my discount code isn't working?",
        "Can you guide me through the process of resolving a non-functional discount code?",
        "How can I troubleshoot issues with my discount code during checkout?",
        "Do you have any recommendations for fixing a discount code that's not working?",
        "Are there common reasons why a discount code might not work?",
        "Can I get assistance on resolving issues with my non-working discount code?",
        "What should I check if my discount code is not being accepted?",
        "Are there specific conditions that discount codes must meet to work?",
        "Can you provide a checklist for diagnosing a malfunctioning discount code?",
        "What steps should I follow to resolve a discount code that's not functioning?"
      ],
      "responses": [
        "If your discount code is not working, please double-check the terms and conditions associated with the code. If the issue persists, contact our customer support team for assistance."
      ]
    },
    {
      "tag": "finalSale",
      "patterns": [
        "What does it mean when a product is labeled as 'final sale'?",
        "Can you explain the implications of a product being marked as 'final sale'?",
        "What are the characteristics of items categorized as 'final sale'?",
        "Are there specific conditions associated with final sale items?",
        "Can you elaborate on the non-returnable nature of final sale products?",
        "Do all final sale items have a no-return policy?",
        "What restrictions apply to products designated as 'final sale'?",
        "Is there a no-refund policy for items labeled as 'final sale'?",
        "Can final sale items be exchanged for other products?",
        "What should I consider before purchasing a final sale item?"
      ],
      "responses": [
        "Final sale items are usually non-returnable and non-refundable. Please review the product description or contact our customer support team to confirm the return eligibility for specific items."
      ]
    },
    {
      "tag": "installation",
      "patterns": [
        "Which products qualify for installation services?",
        "Can you provide a list of products eligible for installation services?",
        "How can I identify products that come with installation services?",
        "Are installation services offered for all products or only specific ones?",
        "Is there a section in the product description indicating installation availability?",
        "Do you have a dedicated category for products with installation options?",
        "Can I filter products based on the availability of installation services?",
        "Are there specific criteria for products to be eligible for installation?",
        "What information is provided in the product description regarding installation?",
        "Do all products with installation services include the option for installation?"
      ],
      "responses": [
        "Installation services are available for select products. Please check the product description or contact our customer support team for more information and to request installation services."
      ]
    },
    {
      "tag": "giftMessage",
      "patterns": [
        "What is the process for adding a gift message to my order?",
        "Can you guide me on how to include a personalized message with my order?",
        "Is there a specific section during checkout for adding a gift message?",
        "Do I have the option to add a gift message to any product in my order?",
        "Are there character limits for the gift message?",
        "Can I preview how the gift message will appear on the order?",
        "Is the gift message visible to the recipient?",
        "Can I edit the gift message after placing the order?",
        "Are there any additional charges for including a gift message?",
        "Can I add a gift message after completing the checkout process?"
      ],
      "responses": [
        "Yes, you can add a gift message during the checkout process. There is usually a section where you can enter your personalized message."
      ]

    },
    {
      "tag": "productDemo",
      "patterns": [
        "Do you provide product demonstrations before purchase?",
        "Is there a way to request a product demonstration?",
        "Are product demonstrations available for all products?",
        "Can I schedule a product demonstration with your team?",
        "Where can I find detailed information if product demonstrations are not offered?",
        "Are there alternative resources for understanding product features?",
        "Can I view video demonstrations of your products online?",
        "Do you plan to introduce product demonstrations in the future?",
        "Are there any events or exhibitions where I can see your products in action?",
        "How can I make an informed decision without a product demonstration?"
      ],
      "responses": [
        "We do not currently offer product demonstrations before purchase. However, you can find detailed product descriptions, specifications, and customer reviews on our website."
      ]
    },
    {
      "tag": "invoice",
      "patterns": [
        "Is an invoice automatically included with my order?",
        "Can I request a separate invoice for my order?",
        "How can I obtain an invoice for my purchase?",
        "Is the invoice sent electronically or included in the package?",
        "Are there specific details included in the order invoice?",
        "Can I request an invoice with additional information for business purposes?",
        "Is there any sensitive information on the order invoice?",
        "Do you provide digital copies of the order invoice?",
        "Can I get a copy of the invoice if I misplaced the original?",
        "Is there a specific format for the order invoice?"
      ],
      "responses": [
        "Yes, an invoice is usually included with your order. If you require a separate invoice, please contact our customer support team with your order details."
      ]
    },
    {
      "tag": "returnPackaging",
      "patterns": [
        "Is it possible to return a product if I no longer have the original packaging?",
        "What should I do if I want to return a product but don't have the original packaging?",
        "Are there any specific requirements for returning a product without its original packaging?",
        "Can I still initiate a return if the original packaging is damaged?",
        "Is returning a product without the original packaging subject to any additional fees?",
        "Are there alternative packaging options for returning a product?",
        "Do you provide return packaging for products without their original packaging?",
        "Can I get assistance from customer support if I have concerns about returning a product without its original packaging?",
        "Is there a timeframe within which I should return a product without the original packaging?",
        "What information is required when initiating a return without the original packaging?"
      ],
      "responses": [
        "While returning a product in its original packaging is preferred, you can still initiate a return without it. Contact our customer support team for guidance in such cases."
      ]
    },
    {
      "tag": "preOrderInStock",
      "patterns": [
        "Can I place an order with a mix of pre-order and in-stock items?",
        "Is there a specific process for ordering a mix of pre-order and in-stock items?",
        "How are shipping costs calculated for orders with pre-order and in-stock items?",
        "Can I modify a pre-order item in an existing order?",
        "Are there any restrictions on the quantity of pre-order items in a single order?",
        "Can I prioritize the shipment of in-stock items in a mixed-order?",
        "What happens if the release date of a pre-order item changes after I place the order?",
        "Can I cancel a pre-order item in a mixed-order?",
        "Is there a separate checkout process for pre-order items?",
        "Can I add additional in-stock items to a pre-order without affecting the release date?"
      ],
      "responses": [
        "Yes, you can place an order with a mix of pre-order and in-stock items. However, please note that the entire order will be shipped once all items are available."
      ]
    },
    {
      "tag": "returnGift",
      "patterns": [
        "Can I return a product purchased as a gift?",
        "Is there a different process for returning gifts compared to regular purchases?",
        "Can I receive a refund for a gifted product?",
        "Do I need the original payment method used for the gift purchase to process a return?",
        "Is there a time limit for returning gifted items?",
        "Can I exchange a gifted item for a different product?",
        "Is the return process different for items purchased as gifts?",
        "What happens if the gift recipient wants to return the item?",
        "Can I receive store credit for a returned gifted product?",
        "Are there any restrictions on returning gift items without a receipt?"
      ],
      "responses": [
        "Yes, you can return a product purchased as a gift. However, refunds will typically be issued to the original payment method used for the purchase."
      ]
    },
    {
      "tag": "size",
      "patterns": [
        "If a product is not available in my size, what options do I have?",
        "Can I request notifications when a specific size is restocked?",
        "Are there alternative sizing options for products temporarily out of stock?",
        "Can I find similar products in my size if the one I want is unavailable?",
        "Do you restock sizes regularly for popular products?",
        "Are there any discounts or promotions for products currently out of stock in my size?",
        "Can I pre-order a product in my size if it's temporarily unavailable?",
        "How long does it usually take to restock products in specific sizes?",
        "Is there a waitlist for specific sizes of popular products?",
        "Are there any alternative products available if the desired size is out of stock?"
      ],
      "responses": [
        "If a product is not available in your size, it may be temporarily out of stock. Please check back later or sign up for size notifications."
      ]
    },
    {
      "tag": "returnDiscount", 
      "patterns": [
        "What is the return process for products purchased with a discount code?",
        "Can I receive a cash refund for products bought with a discount code?",
        "Is there a time limit for returning items purchased with a discount code?",
        "Can I exchange a product purchased with a discount code for a different item?",
        "Do discount code returns result in a new discount code or store credit?",
        "Is there a difference in the return process for discount code purchases?",
        "What happens if the discount code used for the purchase is no longer available?",
        "Can I use store credit from a discount code return on future purchases?",
        "Are there any restrictions on returning products bought with a discount code?",
        "Can I return the discount code itself for a refund?"
      ],
      "responses": [
        "Yes, you can return a product purchased with a discount code. The refund will be processed based on the amount paid after the discount."
      ]
    },
    {
      "tag": "customeOrder",
      "patterns": [
        "Do you accept requests for custom orders or personalized products?",
        "Is there a specific reason why custom orders or personalized products are not offered?",
        "Are there any plans to introduce custom order options in the future?",
        "Can I make special requests for existing products to be personalized?",
        "Is there a possibility of adding custom engraving or personalization to products?",
        "Are there any exceptions for certain products to be personalized on request?",
        "What alternatives do you suggest if custom orders are not available?",
        "Can I request modifications to existing products to suit my preferences?",
        "Do you collaborate with customers for unique product designs?",
        "Are there any limitations on product personalization even if not custom ordered?"
      ],
      "responses": [
        "We do not currently offer custom orders or personalized products. Please explore the available products on our website."
      ]
    },
    {
      "tag": "temporarilyUnavailable",
      "patterns": [
        "How can I order a product listed as 'temporarily unavailable'?",
        "Can I be notified when a 'temporarily unavailable' product is restocked?",
        "Are there any waitlists for 'temporarily unavailable' items?",
        "Can I reserve a 'temporarily unavailable' product for future purchase?",
        "Do you restock 'temporarily unavailable' products regularly?",
        "Are there any discounts or promotions for 'temporarily unavailable' items?",
        "Can I find 'temporarily unavailable' products through authorized retailers?",
        "Is there a specific time frame for 'temporarily unavailable' products to be restocked?",
        "What factors contribute to products being labeled as 'temporarily unavailable'?",
        "Can I get information on past customer reviews for 'temporarily unavailable' products?"
      ],
      "responses": [
        "If a product is listed as 'temporarily unavailable,' it is out of stock but may be restocked in the future. Please check back later or sign up for notifications."
      ]
    },
    {
      "tag": "defect",
      "patterns": [
        "What is the process for returning a product damaged due to improper use?",
        "Can I get a refund for a product damaged because of improper use?",
        "Is there a difference in the return process for products damaged due to improper use?",
        "Do you provide guidelines for proper use to avoid potential damage?",
        "Can I request a repair or replacement for a product damaged by improper use?",
        "What constitutes 'improper use' for product damage claims?",
        "Are there any exceptions for returning products damaged due to improper use?",
        "Is there a specific timeframe for reporting products damaged due to improper use?",
        "Do you offer assistance in preventing damage through proper product use?",
        "Are there any educational resources on product use to prevent damage?"
      ],
      "responses": [
        "Our return policy generally covers products that are defective or damaged upon arrival. Damage due to improper use may not be eligible for a return. Please contact our customer support team for assistance."
      ]
    },
    {
      "tag": "onHold",
      "patterns": [
        "What does it mean if a product is listed as 'on hold'?",
        "Are there any specific reasons for placing products 'on hold' temporarily?",
        "Can I receive notifications when a product listed as 'on hold' becomes available?",
        "Is there a specific time frame for products listed as 'on hold' to be restocked?",
        "Can I reserve a product listed as 'on hold' for future purchase?",
        "Do products listed as 'on hold' have any special considerations for ordering?",
        "Can I pre-order products listed as 'on hold'?",
        "What alternatives do you recommend if a product listed as 'on hold' is essential for me?",
        "Are there any promotions or discounts associated with products listed as 'on hold'?",
        "Can I still explore customer reviews for products listed as 'on hold'?"
      ],
      "responses": [
        "If a product is listed as 'on hold,' it is temporarily unavailable for purchase. Please check back later or sign up for notifications when it becomes available."
      ]
    },
    {
      "tag": "returnNoReceipt",
      "patterns": [
        "Is it possible to return a product without the original receipt?",
        "Are there any alternative proofs of purchase accepted for returns?",
        "Do you have a specific process for returns without the original receipt?",
        "Can I provide other details in place of the original receipt for returns?",
        "Is there a timeframe within which returns without the original receipt must be initiated?",
        "Do returns without the original receipt follow the same refund process?",
        "Are there any exceptions or limitations for returns without the original receipt?",
        "Can I request a copy of the original receipt for return purposes?",
        "What information is required when initiating a return without the original receipt?",
        "Are there any differences in the return policy for returns without the original receipt?"
      ],
      "responses": [
        "While a receipt is preferred for returns, we may be able to assist you without it. Please contact our customer support team for further guidance."
      ]
    },
    {
      "tag": "limitedEdition",
      "patterns": [
        "Can limited edition products be restocked upon customer request?",
        "What defines the availability period for limited edition products?",
        "Are there any exceptions for restocking limited edition items?",
        "Can I be notified if a limited edition product is restocked?",
        "Is there a limit to the number of limited edition products I can order?",
        "Are there any advantages to ordering limited edition products during restocks?",
        "Can I make special requests for limited edition products to be restocked?",
        "How often do limited edition products get restocked?",
        "Can I reserve a limited edition product for future purchase?",
        "Do limited edition products have a different return policy compared to regular items?"
      ],
      "responses": [
        "Once a limited edition product is sold out, it may not be restocked. Limited edition items are available for a limited time only, so we recommend purchasing them while they are available."
      ]
    },
    {
      "tag": "discontinued",
      "patterns": [
        "If a product is listed as 'discontinued' but still visible on the website, can it be ordered?",
        "Is there a specific reason for a 'discontinued' product still being visible on the website?",
        "Can I request clarification on the status of a 'discontinued' product?",
        "Is there a possibility of a 'discontinued' product being available through other channels?",
        "Can I be notified if a 'discontinued' product becomes available again?",
        "Do 'discontinued' products have any special considerations for ordering?",
        "Are there any promotions or discounts associated with 'discontinued' products?",
        "Is there any information available on why a product is labeled 'discontinued'?",
        "Can I explore customer reviews for 'discontinued' products?",
        "Is there a way to confirm the accuracy of a 'discontinued' label on a product?"
      ],
      "responses": [
        "If a product is listed as 'discontinued' but still visible on the website, it may be an error. Please contact our customer support team for clarification."
      ]
    },
    {
      "tag": "clearance",
      "patterns": [
        "Can clearance or final sale items be returned?",
        "Are there any exceptions or conditions for returning clearance or final sale items?",
        "Is there a specific process for returning clearance or final sale items?",
        "Do clearance or final sale items follow the same refund process as regular items?",
        "Can I exchange a clearance or final sale item for a different product?",
        "Are there any advantages to purchasing clearance or final sale items?",
        "Can I request additional information on clearance or final sale items before purchase?",
        "Is there a time limit for initiating returns on clearance or final sale items?",
        "Do clearance or final sale items have a different warranty or guarantee?",
        "Can I find customer reviews for clearance or final sale items?"
      ],
      "responses": [
        "Clearance or final sale items are typically non-returnable and non-refundable. Please review the product description or contact our customer support team for more information."
      ]
    },
    {
      "tag": "notListed",
      "patterns": [
        "What should I do if the product I want is not listed on your website?",
        "Is there a process for requesting products not currently available on your website?",
        "Are there any channels for suggesting new products to be added to your inventory?",
        "Can I expect new products to be added to your website regularly?",
        "Is there a way to receive notifications when new products are added to your inventory?",
        "How often do you update your product listings?",
        "Are there specific criteria for selecting products to be listed on your website?",
        "Can I provide feedback on the products I would like to see on your website?",
        "Do you have plans to expand your product offerings in the future?",
        "Is there a possibility of products not currently listed being added based on customer requests?"
      ],
      "responses": [
        "If a product is not listed on our website, it may not be available for purchase. We recommend exploring the available products or contacting our customer support team for further assistance."
      ]
    },
    {
      "tag": "returnBundle",
      "patterns": [
        "How does the return policy apply to products purchased as part of a bundle or set?",
        "Are there specific conditions for returning individual items from a bundle or set?",
        "Can I return a single item from a bundle or set, or is it necessary to return the entire set?",
        "Do bundled or set items have a different return process compared to individual items?",
        "Is there a time limit for initiating returns on bundled or set items?",
        "Can I exchange an item from a bundle or set for a different product?",
        "Are there any advantages or discounts associated with purchasing bundled or set items?",
        "Can I find customer reviews specifically for bundled or set items?",
        "Are there limitations on the types of products available in bundles or sets?",
        "Do bundled or set items have a different warranty or guarantee?"
      ],
      "responses": [
        "If a product was purchased as part of a bundle or set, the return policy may vary. Please refer to the specific terms and conditions or contact our customer support team for further guidance."
      ]
    },
    {
      "tag": "preorder",
      "patterns": [
        "If a product is listed as 'coming soon' and available for pre-order, can I place an order?",
        "How does the pre-order process work for products listed as 'coming soon'?",
        "Can I reserve a product through pre-order and pay later?",
        "Is there an estimated time for shipping pre-ordered products?",
        "Do pre-ordered products follow the same shipping policies as in-stock items?",
        "Can I cancel a pre-order if I change my mind before it's shipped?",
        "Are there any advantages to ordering products on pre-order?",
        "Is there a limit to the number of items I can pre-order?",
        "Can I use promotional codes for products placed on pre-order?",
        "Are there any special considerations for returns on pre-ordered items?"
      ],
      "responses": [
        "If a product is listed as 'coming soon' and available for pre-order, you can place an order to secure your item before it becomes available."
      ]
    },
    {
      "tag": "damagedshipping",
      "patterns": [
        "What should I do if the product I receive is damaged due to mishandling during shipping?",
        "Is there a specific process for handling products damaged during shipping?",
        "Can I expect assistance if my product is damaged due to mishandling during shipping?",
        "Are there any specific steps to follow for returning a product damaged during shipping?",
        "Is there a timeframe for reporting products damaged during shipping?",
        "Do I need to provide any evidence or documentation for a product damaged during shipping?",
        "Are there any specific guidelines for packaging and returning a damaged product?",
        "How long does it take to process the return and replacement of a product damaged during shipping?",
        "Is there any cost associated with returning a product damaged during shipping?",
        "Can I choose a replacement or a refund for a product damaged during shipping?"
      ],
      "responses": [
        "If your product was damaged due to mishandling during shipping, please contact our customer support team immediately. We will assist you with the necessary steps for return and replacement."
      ]
    },
    {
      "tag": "outofstock",
      "patterns": [
        "Do you offer reservations for out-of-stock products?",
        "Is there a way to reserve a specific product that is currently out of stock?",
        "Can I expect priority access to out-of-stock products through reservations?",
        "How does the reservation system work for out-of-stock products?",
        "Is there a waiting list for reserved out-of-stock products?",
        "Can I cancel a reservation for an out-of-stock product if I change my mind?",
        "Are there any advantages to reserving out-of-stock products?",
        "How do I know when a reserved out-of-stock product becomes available for purchase?",
        "Is there a limit to the number of products I can reserve at a time?",
        "Can I use promotional codes for reserved out-of-stock products?"
      ],
      "responses": [
        "We do not offer reservations for out-of-stock products. However, you can sign up for product notifications to be alerted when the item becomes available again."
      ]
    },
    {
      "tag": "backorder",
      "patterns": [
        "If a product is listed as 'pre-order' and available for backorder, can I still place an order?",
        "How does the pre-order and backorder process work for products listed as 'pre-order'?",
        "Is there an estimated time for shipping pre-ordered and backordered products?",
        "Do pre-ordered and backordered products follow the same shipping policies as in-stock items?",
        "Can I cancel a pre-order or backorder if I change my mind before it's shipped?",
        "Are there any advantages to ordering products on pre-order or backorder?",
        "Is there a limit to the number of items I can pre-order or backorder?",
        "Can I use promotional codes for products placed on pre-order or backorder?",
        "Are there any special considerations for returns on pre-ordered or backordered items?"
      ],
      "responses": [
        "If a product is listed as 'pre-order' and available for backorder, you can place an order to secure your item. The product will be shipped once it becomes available."
      ]
    },
    {
      "tag": "storeCredit",
      "patterns": [
        "If I purchase a product with store credit, can I still return it?",
        "Is there a different process for returning products purchased with store credit?",
        "Can I expect a refund in the form of store credit for products purchased with store credit?",
        "How does the return process differ for products bought with store credit?",
        "Can I combine store credit with other payment methods for a purchase?",
        "Is there a limit to the amount of store credit that can be used for a return?",
        "Can I transfer store credit to someone else?",
        "Are there any restrictions on the products I can purchase with store credit?",
        "Do store credit refunds have an expiration date?",
        "Can I request a cash refund for a product purchased with store credit?"
      ],
      "responses": [
        "Yes, you can return a product purchased with store credit. The refund will be issued in the form of store credit, which you can use for future purchases."
      ]
    },
    {
      "tag": "restock",
      "patterns": [
        "How likely is it for a product currently out of stock to be restocked?",
        "Is there a specific restocking schedule for out-of-stock products?",
        "Can I request restocking for a specific product currently out of stock?",
        "Are there any criteria for determining which out-of-stock products are restocked?",
        "Do restocked products follow the same pricing as before going out of stock?",
        "Can I reserve an out-of-stock product once it is restocked?",
        "Is there a notification system for alerting customers when products are restocked?",
        "Are there any promotions or discounts associated with restocked products?",
        "Can I pre-order an out-of-stock product before it is restocked?",
        "Do restocked products have the same warranty or guarantee as regular items?"
      ],
      "responses": [
        "We strive to restock popular products whenever possible. Please sign up for product notifications to be informed when the item becomes available again."
      ]
    },
    {
      "tag": "soldout", 
      "patterns": [
        "If a product is listed as 'sold out' but available for pre-order, can I still place an order?",
        "How does the ordering process work for products listed as 'sold out' but available for pre-order?",
        "Can I reserve a product through pre-order and pay later for items listed as 'sold out'?",
        "Is there an estimated time for shipping pre-ordered products listed as 'sold out'?",
        "Do pre-ordered items listed as 'sold out' follow the same shipping policies as in-stock items?",
        "Can I cancel a pre-order if I change my mind before it's shipped for products listed as 'sold out'?",
        "Are there any advantages to ordering products on pre-order for items listed as 'sold out'?",
        "Is there a limit to the number of items I can pre-order for products listed as 'sold out'?",
        "Can I use promotional codes for products placed on pre-order for items listed as 'sold out'?",
        "Are there any special considerations for returns on pre-ordered items listed as 'sold out'?"
      ],
      "responses": [
        "If a product is listed as 'sold out' but available for pre-order, you can place an order to secure your item. The product will be shipped once it becomes available."
      ]
    },
    {
      "tag": "returnGiftCard",
      "patterns": [
        "Can I return a product if it was purchased with a promotional gift card?",
        "Is there a different process for returning products purchased with a promotional gift card?",
        "Can I expect a refund in the form of store credit for products purchased with a promotional gift card?",
        "How does the return process differ for products bought with a promotional gift card?",
        "Can I combine a promotional gift card with other payment methods for a purchase?",
        "Is there a limit to the amount of a promotional gift card that can be used for a return?",
        "Can I transfer a promotional gift card to someone else?",
        "Are there any restrictions on the products I can purchase with a promotional gift card?",
        "Do promotional gift card refunds have an expiration date?",
        "Can I request a cash refund for a product purchased with a promotional gift card?"
      ],
      "responses": [
        "Yes, you can return a product purchased with a promotional gift card. The refund will be issued in the form of store credit or a new gift card."
      ]
    },
    {
      "tag": "colorproduct",
      "patterns": [
        "If a product is not currently available in my preferred color, can I still request it?",
        "What options are available if the product is not offered in my preferred color?",
        "Is there a waiting list for preferred colors of out-of-stock products?",
        "How often do preferred colors for products come back in stock?",
        "Can I be notified when the preferred color of a product becomes available?",
        "Are there any alternatives or similar products available in my preferred color?",
        "Is there a specific timeline for restocking products in preferred colors?",
        "Do products in preferred colors follow the same pricing as regular items?",
        "Can I pre-order a product in my preferred color before it becomes available?",
        "Is there a limit to the number of products I can request in my preferred color?"
      ],
      "responses": [
        "If a product is not available in your preferred color, it may be temporarily out of stock. Please check back later or sign up for color notifications."
      ]
    },
    {
      "tag": "comingsoon",
      "patterns": [
        "If a product is listed as 'coming soon' but not available for pre-order, can I still order it?",
        "How does the ordering process work for products listed as 'coming soon' and not available for pre-order?",
        "Can I reserve a product through pre-order and pay later for items listed as 'coming soon'?",
        "Is there an estimated time for shipping products listed as 'coming soon'?",
        "Do products listed as 'coming soon' follow the same shipping policies as in-stock items?",
        "Can I cancel a pre-order if I change my mind before it's shipped for products listed as 'coming soon'?",
        "Are there any advantages to ordering products on pre-order for items listed as 'coming soon'?",
        "Is there a limit to the number of items I can pre-order for products listed as 'coming soon'?",
        "Can I use promotional codes for products placed on pre-order for items listed as 'coming soon'?",
        "Are there any special considerations for returns on pre-ordered items listed as 'coming soon'?"
      ],
      "responses": [
        "If a product is listed as 'coming soon' but not available for pre-order, you will need to wait until it is officially released and becomes available for purchase."
      ]
    },
    {
      "tag": "returnPromotional",
      "patterns": [
        "Can I return a product if it was purchased during a promotional event?",
        "Is there a different process for returning products purchased during a promotional event?",
        "Can I expect a refund in the form of store credit for products purchased during a promotional event?",
        "How does the return process differ for products bought during a promotional event?",
        "Can I combine promotional discounts with other payment methods for a purchase?",
        "Is there a limit to the amount of promotional discounts that can be used for a return?",
        "Can I transfer promotional discounts to someone else?",
        "Are there any restrictions on the products I can purchase with promotional discounts?",
        "Do promotional discount refunds have an expiration date?",
        "Can I request a cash refund for a product purchased during a promotional event?"
      ],
      "responses": [
        "Yes, you can return a product purchased during a promotional event. The refund will be processed based on the amount paid after any applicable discounts."
      ]
    }
]

# Insert data into the FAQ table
for intent in intents:
    question = intent['patterns'][0]  # Assuming the first pattern as the question
    answer = intent['responses'][0]  # Assuming the first response as the answer
    tag = intent['tag']
    
    # Insert into FAQ table
    cursor.execute("INSERT INTO FAQ (question, answer, tag) VALUES (%s, %s, %s)", (question, answer, tag))

# Commit changes and close connection
conn.commit()
conn.close()