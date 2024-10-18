import stripe
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, get_user_model
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm,LoginForm
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse
from django.http import JsonResponse
from django.contrib.auth import logout
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from subscription.models import Subscription ,SubscriptionPlan, Payment
from datetime import timedelta
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags

User = get_user_model()

stripe.api_key = settings.STRIPE_SECRET_KEY

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)

            if user is not None:
                if user.is_active:  # Check if the user's email is verified
                    login(request, user)
                    messages.success(request, 'Logged in successfully!')
                    return redirect('dashboard')  # Redirect to the desired page
                else:
                    messages.error(request, 'Your account is not active. Please verify your email address.')
            else:
                messages.error(request, 'Invalid username or password')
    else:
        form = LoginForm()
    
    return render(request, 'auth/login.html', {'form': form})


def register_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            try:
                # Save the user but set `is_active` to False until email verification
                user = form.save(commit=False)
                user.is_active = False  # Deactivate the account until email verification
                user.save()

                # Get the selected subscription plan from the form
                subscription_plan = form.cleaned_data.get('subscription_plan')

                # Create a subscription for the user
                subscription = Subscription.objects.create(
                    user=user,
                    plan=subscription_plan,
                    status='pending'  # You can set the status to active or pending, depending on your logic
                )

                # Generate the verification token
                token = default_token_generator.make_token(user)
                uid = urlsafe_base64_encode(force_bytes(user.pk))

                # Get the current site's domain
                current_site = get_current_site(request)
                domain = current_site.domain

                # Build the verification link
                verification_link = reverse('activate', kwargs={'uidb64': uid, 'token': token})
                verification_url = f"http://{domain}{verification_link}"

                # Email content
                subject = 'Verify your email address'
                from_email = settings.EMAIL_HOST_USER
                recipient_list = [user.email]

                # Render the HTML email template
                html_content = render_to_string('auth/verification_email.html', {
                    'user': user,
                    'verification_url': verification_url,
                })

                # Strip HTML tags for plain text version (optional but recommended)
                text_content = strip_tags(html_content)

                # Send HTML email
                email = EmailMultiAlternatives(subject, text_content, from_email, recipient_list)
                email.attach_alternative(html_content, "text/html")
                email.send()

                # Display success message
                messages.success(request, 'Account created! Please check your email to verify your account.')
                return redirect('login')

            except Exception as e:
                # Handle errors in subscription or user creation process
                messages.error(request, 'There was an issue processing your registration. Please try again.')
                # Optionally, you can log the error message e to keep track of errors
        else:
            messages.error(request, 'Please correct the errors in the form.')
    else:
        form = SignUpForm()

    return render(request, 'auth/register.html', {'form': form})

def activate(request, uidb64, token):
    try:
        # Decode the user ID from the base64 encoded uid
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    # Check if the token is valid and activate the user
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()

        # Get the user's subscription plan
        subscription = user.subscriptions.last()  # Assuming last subscription is the one selected during signup

        if subscription and subscription.plan:
            # Check if the plan is free
            if subscription.plan.price == 0:  # Assuming free plans have price = 0
                subscription.status = 'active'
                subscription.start_date = timezone.now()
                subscription.end_date = timezone.now() + timezone.timedelta(days=subscription.plan.duration_days)
                subscription.save()

                # Log in the user and redirect to the dashboard
                messages.success(request, f"Welcome {user.username}, your account has been verified with the free {subscription.plan.name} plan!")
                return redirect('dashboard')
            
            else:
                # Handle paid plans - redirect to Stripe Checkout
                try:
                    checkout_session = stripe.checkout.Session.create(
                        payment_method_types=['card'],
                        line_items=[
                            {
                                'price': subscription.plan.stripe_price_id,  # Use the Stripe price ID for subscriptions
                                'quantity': 1,
                            },
                        ],
                        mode='subscription',
                        success_url=request.build_absolute_uri(
                            reverse('payment_success') + f"?session_id={{CHECKOUT_SESSION_ID}}&user_id={user.id}&plan_id={subscription.plan.id}"
                        ),
                        cancel_url=request.build_absolute_uri(reverse('payment_cancel')),
                    )
                    return redirect(checkout_session.url, code=303)

                except stripe.error.StripeError as e:
                    messages.error(request, f"Error creating Stripe Checkout session: {e.user_message}")
                    return redirect('login')

        else:
            messages.error(request, "No subscription plan was selected during registration.")
            return redirect('login')

    else:
        messages.error(request, 'The activation link is invalid or has expired.')
        return render(request, 'auth/activation_invalid.html')



def payment_success_view(request):
    user_id = request.GET.get('user_id')
    plan_id = request.GET.get('plan_id')
    session_id = request.GET.get('session_id')  # Stripe session ID to track the payment

    # Fetch the user and plan details from the database
    user = get_object_or_404(User, id=user_id)
    plan = get_object_or_404(SubscriptionPlan, id=plan_id)

    # Log the user in, if not already (optional)
    login(request, user)

    # Cancel any existing active subscriptions for this user
    active_subscriptions = Subscription.objects.filter(user=user, status='active')
    for active_subscription in active_subscriptions:
        active_subscription.status = 'canceled'
        active_subscription.end_date = timezone.now()  # Mark the end date for the canceled subscription
        active_subscription.save()

    # Create the new subscription for the user
    new_subscription = Subscription.objects.create(
        user=user,
        plan=plan,
        start_date=timezone.now(),
        end_date=timezone.now() + timedelta(days=plan.duration_days),
        status='active'
    )

    # Log the payment details in the Payment model
    Payment.objects.create(
        subscription=new_subscription,
        amount=plan.price,  # Assuming SubscriptionPlan has a price field
        stripe_payment_intent_id=session_id,  # Store the Stripe session/payment intent ID for reference
        status='completed'
    )

    # Send payment confirmation email
    subject = 'Payment Confirmation'
    from_email = settings.EMAIL_HOST_USER
    recipient_list = [user.email]

    # Render the HTML email template for payment confirmation
    html_content = render_to_string('auth/payment_confirmation_email.html', {
        'user': user,
        'plan': plan,
        'amount': plan.price,
        'subscription': new_subscription,
    })

    # Strip HTML tags for the plain text version
    text_content = strip_tags(html_content)

    # Send the email
    email = EmailMultiAlternatives(subject, text_content, from_email, recipient_list)
    email.attach_alternative(html_content, "text/html")
    email.send()

    # Notify the user about their successful subscription
    messages.success(request, f"Thank you for your purchase, {user.username}! You have successfully subscribed to the {plan.name} plan.")

    # Redirect to a success page or render the success template
    return render(request, 'auth/success.html')

def payment_cancel_view(request):
    messages.warning(request, "Your payment has been canceled.")
    return redirect('register')

@csrf_exempt
def stripe_webhook_view(request):
    payload = request.body
    sig_header = request.META['HTTP_STRIPE_SIGNATURE']
    event = None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        return JsonResponse({'status': 'invalid payload'}, status=400)
    except stripe.error.SignatureVerificationError as e:
        return JsonResponse({'status': 'invalid signature'}, status=400)

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        # Fulfill the purchase
        user_id = session['metadata']['user_id']
        plan_id = session['metadata']['plan_id']
        user = User.objects.get(id=user_id)
        plan = SubscriptionPlan.objects.get(id=plan_id)
        
        # Create a subscription record
        Subscription.objects.create(
            user=user,
            plan=plan,
            start_date=timezone.now(),
            end_date=timezone.now() + timezone.timedelta(days=plan.duration_days)
        )
        
        # Additional logic here
    return JsonResponse({'status': 'success'})

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')  # Redirect to the login page or any other page after logout