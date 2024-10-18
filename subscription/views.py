from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import SubscriptionPlan, Subscription, Payment
from django.contrib import messages
from datetime import timedelta
import stripe
from django.conf import settings
from django.urls import reverse

# Configure the Stripe API key
stripe.api_key = settings.STRIPE_SECRET_KEY

@login_required
def subscription_list(request):
    # View to display available subscription plans
    plans = SubscriptionPlan.objects.filter(active=True)

    context = {
        'plans': plans
    }
    return render(request, 'subscriptions/subscription_list.html', context)

@login_required
def change_subscription(request):
    # Get the user's current active subscription
    current_subscription = Subscription.objects.filter(user=request.user, status='active').first()

    # Check if the user has an active subscription
    if not current_subscription:
        messages.error(request, "You don't have an active subscription to change.")
        return redirect('subscriptions:subscription_list')

    if request.method == 'POST':
        new_plan_id = request.POST.get('new_plan_id')
        new_plan = get_object_or_404(SubscriptionPlan, id=new_plan_id, active=True)

        # Check if the new plan is the same as the current one
        if new_plan == current_subscription.plan:
            messages.error(request, "You are already subscribed to this plan.")
            return redirect('dashboard:dashboard_home')

        # Calculate the new end date based on the duration of the new plan
        new_end_date = timezone.now() + timedelta(days=new_plan.duration_days)

        # For free plans, change the subscription directly without redirecting to Stripe
        if new_plan.free:
            current_subscription.plan = new_plan
            current_subscription.start_date = timezone.now()  # Set the new start date for the new plan
            current_subscription.end_date = new_end_date  # Set the end date based on the new plan's duration
            current_subscription.save()

            messages.success(request, f"Your subscription has been changed to the {new_plan.name} plan, valid until {new_end_date.strftime('%B %d, %Y')}.")
            return redirect('dashboard')
        else:
            # For paid plans, redirect to Stripe checkout
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                customer_email=request.user.email,
                line_items=[
                                {
                                    'price': new_plan.stripe_price_id,  # Use the Stripe price ID for subscriptions
                                    'quantity': 1,
                                },
                            ],
                mode='subscription',
               success_url=request.build_absolute_uri(
                                reverse('payment_success') + f"?session_id={{CHECKOUT_SESSION_ID}}&user_id={request.user.id}&plan_id={new_plan.id}"
                            ),
                            cancel_url=request.build_absolute_uri(reverse('payment_cancel')),
                metadata={
                    'user_id': request.user.id,
                    'plan_id': new_plan.id,
                    'current_subscription_id': current_subscription.id
                }
            )

            # Create a pending payment record
            

            return redirect(session.url, code=303)

    # Get all plans except the current one
    plans = SubscriptionPlan.objects.filter(active=True).exclude(id=current_subscription.plan.id)
    context = {
        'plans': plans,
        'current_subscription': current_subscription,
    }
    return render(request, 'subscriptions/change_subscription.html', context)

@login_required
def subscribe(request, plan_id):
    plan = get_object_or_404(SubscriptionPlan, id=plan_id, active=True)
    current_subscription = Subscription.objects.filter(user=request.user, status='active', plan=plan).first()

    # Check if the user already has an active subscription to this plan
    if current_subscription:
        messages.error(request, "You already have an active subscription.")
        return redirect('subscription_list')

    try:
        # Calculate the end date based on the plan's details
        if plan.free:
            # For free plans, use trial days to determine the end date
            end_date = timezone.now() + timedelta(days=plan.trial_days)
            # Create the subscription without redirecting to Stripe
            subscription = Subscription.objects.create(
                user=request.user,
                plan=plan,
                start_date=timezone.now(),
                end_date=end_date,
                status='active'
            )
            messages.success(request, f"You have subscribed to the {plan.name} plan.")
            return redirect('dashboard')
        else:
            # Create a pending subscription record before Stripe payment
            subscription = Subscription.objects.create(
                user=request.user,
                plan=plan,
                start_date=timezone.now(),
                end_date=timezone.now() + timedelta(days=plan.duration_days),
                status='pending'  # Mark as pending until payment is successful
            )

            # For paid plans, create a Stripe checkout session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                customer_email=request.user.email,
                line_items=[
                    {
                        'price': plan.stripe_price_id,  # Use the Stripe price ID for subscriptions
                        'quantity': 1,
                    },
                ],
                mode='subscription',  # Use 'subscription' mode for recurring payments
                success_url=request.build_absolute_uri(
                                reverse('payment_success') + f"?session_id={{CHECKOUT_SESSION_ID}}&user_id={request.user.id}&plan_id={plan.id}"
                            ),
                cancel_url=request.build_absolute_uri(reverse('payment_cancel')),
                metadata={
                    'subscription_id': subscription.id,
                    'user_id': request.user.id,
                    'plan_id': plan.id
                }
            )

            # Create a pending payment record
           

            return redirect(session.url, code=303)

    except stripe.error.StripeError as e:
        messages.error(request, f"Stripe error: {e.user_message}")
    except Exception as e:
        messages.error(request, str(e))
    return redirect('subscription_list')


@login_required
def cancel_subscription(request, subscription_id):
    # View to cancel the subscription
    subscription = get_object_or_404(Subscription, id=subscription_id, user=request.user)
    
    if subscription.status == 'active':
        subscription.status = 'canceled'
        subscription.end_date = timezone.now()
        subscription.save()
        messages.success(request, "Your subscription has been canceled.")
    else:
        messages.error(request, "Subscription is already canceled or expired.")

    return redirect('dashboard')

@login_required
def payment_history(request):
    # View to display user's payment history
    payments = Payment.objects.filter(subscription__user=request.user).order_by('-date')
    return render(request, 'subscriptions/payment_history.html', {'payments': payments})

