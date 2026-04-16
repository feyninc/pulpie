Medusa Checkout Flow: Step-by-Step Guide to Building a Complete E-Commerce Checkout

Building an e-commerce storefront requires implementing a robust checkout flow that handles cart management, shipping, and payment processing. Medusa.js provides a comprehensive set of [Store APIs](https://docs.medusajs.com/api/store) that make this process straightforward. In this article, we'll explore the complete Medusa checkout flow by walking through each API endpoint required to take a customer from an empty cart to a completed order.

Understanding the Medusa Checkout Process

The Medusa checkout flow consists of seven essential steps:

**Cart Creation** - Initialize a new shopping cart

**Add Items** - Add products to the cart

**Set Shipping Address** - Configure delivery information

**Select Shipping Method** - Choose delivery options

**Payment Collection** - Initialize payment processing

**Payment Session** - Set up payment provider

**Complete Order** - Finalize the purchase

Each step builds upon the previous one, creating a seamless user experience while maintaining data integrity throughout the process.

Prerequisites and Setup

Before diving into the implementation, ensure you have:

A running Medusa v2 application

Valid publishable API key

Product variants, regions, and shipping options configured

Payment provider set up (we'll use the system default)

`curl` and `jq` installed for making API requests and parsing JSON responses

`curl` and `jq` Installation

`# macOS (using Homebrew)   brew install  curl jq  # Ubuntu/Debian   sudo  apt update sudo  apt  install  curl jq  # Windows (using Chocolatey)   choco install  curl jq`

Environment Variables

Set the following variables before running the checkout flow:

`export  BASE_URL ="<http://localhost:9000>" export  PUBLISHABLE_KEY ="pk_<your_publishable_key_here>" export  REGION_ID ="reg_<your_region_id>" export  PRODUCT_VARIANT_ID ="variant_<your_variant_id>" export  SHIPPING_OPTION_ID ="so_<your_shipping_option_id>" export  PAYMENT_PROVIDER_ID ="pp_system_default"`

**Note:** The `CART_ID` and `PAYMENT_COLLECTION_ID` variables should be set during the checkout process after cart creation and payment collection initialization.

Step-by-Step Implementation

1. Create a New Cart

The checkout journey begins with creating a new cart associated with a specific region:

`curl -X POST"$BASE_URL /store/carts" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "region_id": "'$REGION_ID'" }'`

**Key Points:**

Each cart gets a unique identifier used in subsequent API calls

Regions define currency, tax rates, and fulfillment settings

2. Add Products to Cart

Once the cart exists, add product variants with specified quantities:

`curl -X POST"$BASE_URL /store/carts/$CART_ID /line-items" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "variant_id": "'$PRODUCT_VARIANT_ID'", "quantity": 1 }'`

**Key Points:**

Use `variant_id` rather than `product_id` for specific product configurations

Quantities can be updated by calling this endpoint again

Inventory levels are automatically checked during this step

3. Set Shipping Address and Customer Information

Configure the delivery address and customer contact details:

`curl -X POST"$BASE_URL /store/carts/$CART_ID" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "shipping_address": { "first_name": "John", "last_name": "Doe", "address_1": "Nordmarksvej 9", "city": "Billund", "country_code": "dk", "postal_code": "7190", "phone": "1234567890" }, "email": "john.doe@example.com" }'`

**Key Points:**

`country_code` must match the region's available countries

Email is required for order confirmation and customer account association

Billing address can be set separately if different from shipping address

4. Select Shipping Method

Choose from available shipping options for the cart's region and address:

`curl -X POST"$BASE_URL /store/carts/$CART_ID /shipping-methods" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "option_id": "'$SHIPPING_OPTION_ID'" }'`

5. Create Payment Collection

Initialize the payment processing system for the cart:

`curl -X POST"$BASE_URL /store/payment-collections" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "cart_id": "'$CART_ID'" }'`

6. Initialize Payment Session

Set up the payment provider session for processing:

`curl -X POST"$BASE_URL /store/payment-collections/$PAYMENT_COLLECTION_ID /payment-sessions" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "provider_id": "'$PAYMENT_PROVIDER_ID'" }'`

**Key Points:**

Payment providers (Stripe, PayPal, etc.) require specific session initialization

Provider ID must match configured payment providers in your Medusa setup

Session contains payment intent or equivalent for the provider

7. Complete the Order

Finalize the checkout process and create the order:

`curl -X POST"$BASE_URL /store/carts/$CART_ID /complete" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY"`

**Key Points:**

This step validates all cart information and processes payment

Inventory is reserved and order confirmation is triggered

The cart is converted to an immutable order record

Complete Implementation Example

Here's a complete shell script implementing the entire checkout flow:

`#!/bin/bash   # Configuration  BASE_URL ="<http://localhost:9000>" PUBLISHABLE_KEY ="pk_" REGION_ID ="reg_" PRODUCT_VARIANT_ID ="variant_" SHIPPING_OPTION_ID ="so_" PAYMENT_PROVIDER_ID ="pp_system_default"  # 1. Create cart  echo  "Creating cart..." CART_RESPONSE =$(curl -s -X POST"$BASE_URL /store/carts" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "region_id": "'$REGION_ID'" }')   CART_ID =$(echo $CART_RESPONSE | jq -r'.cart.id')  echo  "Cart created with ID:$CART_ID"  # 2. Add item to cart  echo  "Adding item to cart..." curl -s -X POST"$BASE_URL /store/carts/$CART_ID /line-items" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "variant_id": "'$PRODUCT_VARIANT_ID'", "quantity": 1 }' > /dev/null  # 3. Set shipping address and email  echo  "Setting shipping address..." curl -s -X POST"$BASE_URL /store/carts/$CART_ID" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "shipping_address": { "first_name": "John", "last_name": "Doe", "address_1": "Nordmarksvej 9", "city": "Billund", "country_code": "dk", "postal_code": "7190", "phone": "1234567890" }, "email": "john.doe@example.com" }' > /dev/null  # 4. Set shipping method  echo  "Setting shipping method..." curl -s -X POST"$BASE_URL /store/carts/$CART_ID /shipping-methods" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "option_id": "'$SHIPPING_OPTION_ID'" }' > /dev/null  # 5. Create payment collection  echo  "Creating payment collection..." PAYMENT_COLLECTION_RESPONSE =$(curl -s -X POST"$BASE_URL /store/payment-collections" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "cart_id": "'$CART_ID'" }')   PAYMENT_COLLECTION_ID =$(echo $PAYMENT_COLLECTION_RESPONSE | jq -r'.payment_collection.id')  echo  "Payment collection created with ID:$PAYMENT_COLLECTION_ID"  # 6. Initialize payment session  echo  "Initializing payment session..." curl -s -X POST"$BASE_URL /store/payment-collections/$PAYMENT_COLLECTION_ID /payment-sessions" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY" \ -d'{ "provider_id": "'$PAYMENT_PROVIDER_ID'" }' > /dev/null  # 7. Complete the cart  echo  "Completing order..." ORDER_RESPONSE =$(curl -s -X POST"$BASE_URL /store/carts/$CART_ID /complete" \ -H"Content-Type: application/json" \ -H"x-publishable-api-key:$PUBLISHABLE_KEY")   ORDER_ID =$(echo $ORDER_RESPONSE | jq -r'.order.id')  echo  "Order completed successfully:$ORDER_ID"`

Troubleshooting

**Check Service Health**: Test if the service is responding by visiting `http://localhost:9000/store/products` in your browser. You should see a JSON response with available products.

**Validate API Key**: Confirm your publishable API key is correct and has proper permissions for store operations.

**No Products Available**: Verify products exist in your store by checking the `/store/products` endpoint. Ensure products are published and have available variants.

**Invalid Shipping Address**:

 - Validate address format matches the expected country standards
 - Ensure the `country_code` is listed under the region's allowed countries
 - Check `/store/regions/{region_id}` to see available countries for your region

**Shipping Option Not Available**: Verify the shipping option is enabled for the selected region and address. Check `/store/shipping-options` for available options in your region.

**Insufficient Inventory**: Check stock levels before adding items. The API will return inventory errors if requested quantities exceed available stock.

**Payment Provider Issues**:

 - Ensure the payment provider is properly configured in your Medusa setup
 - Handle payment failures gracefully with appropriate error messaging
 - Verify payment provider credentials are valid

Conclusion

The Medusa checkout flow provides a robust foundation for e-commerce applications with its well-structured API endpoints. By following this seven-step process, you can build reliable checkout experiences that handle modern e-commerce requirements.

Key takeaways:

Each step depends on the previous one's completion

Implement proper validation and retry logic

Track cart state throughout the process

Provide user feedback at each step

This approach supports various checkout scenarios including guest flows, multiple payment methods, and subscription purchases.

Frequently Asked Questions (FAQ)

1. What is the Medusa checkout flow?

It is a sequence of Store API interactions (cart creation, line items, addresses, shipping, payment collection, payment session, completion) that transforms a cart into an immutable order.

2. Do I need a payment provider to test locally?

You can create carts, add items, set addresses, and shipping without a live provider. To finalize (step 7) you need at least a test/system payment provider configured (e.g., Stripe test keys or the system default plugin).

3. How do I update or remove a line item?

Use `PATCH /store/carts/{cart_id}/line-items/{line_id}` with a new quantity. Setting quantity to 0 or using `DELETE` on the line item endpoint removes it.

4. How can I apply a discount code or gift card?

Add a discount with `POST /store/carts/{cart_id}/discounts` and a gift card with `POST /store/carts/{cart_id}/gift-cards` providing the respective codes before completing the cart.

5. Can I support guest checkout?

Yes. Provide only an email and shipping address. Later, you can associate the order to a customer account if the user registers with the same email.

6. How do I handle multiple shipping methods or split fulfillment?

Create separate fulfillment sets by leveraging line item shipping profiles and ensure each group has an applicable shipping option. Each added method is posted via `/shipping-methods`. (Advanced customization may require custom fulfillment modules.)

7. What if the payment session fails?

Re-initialize the session by deleting the failed payment session and posting a new one with the same or a different `provider_id`. Always display a retry option in the UI.

8. How do I recover an abandoned cart?

Persist the `cart_id` in client storage. Periodically `GET /store/carts/{cart_id}` to show saved items. Optionally send reminder emails if the user supplied an email but did not complete.

9. How do I localize prices and taxes?

Use regions configured with appropriate currency and tax rates. Selecting the correct `region_id` when creating the cart ensures all price calculations reflect that locale.

10. How can I add custom metadata to a cart or order?

Include a metadata object when updating the cart (`POST /store/carts/{cart_id}`) or extend order handling in a custom module. Metadata persists to the resulting order for analytics or integrations.

!!![Michał Miler]() Michał Miler Senior Software Engineer